"""N7: fit the MANO hand model to the fused hand clouds (fuse.py), per frame.

  python fit_mano.py --shape sessions/2026-11-06_035726 sessions/2026-11-06_035752 sessions/2026-11-06_035809
  python fit_mano.py sessions/2026-11-06_035942

Needs mano/MANO_LEFT_np.pkl / MANO_RIGHT_np.pkl (tools/convert_mano.py) and <session>/fused/hands.npz.
Everything is in cam_A's unit frame, mm (MANO's metres x 1000).

--shape: one MANO shape (betas) for the person, fitted jointly with a pose per frame on --shape-frames
frames from still sessions; saved to mano_shape.json. Without --shape, the saved shape is used.

Per frame: start from the previous frame's pose (the first frame: from LeapC's joints), then minimise
  point-to-surface distance of the fused points (point-to-plane at the nearest MANO vertex, trimmed)
  + weak pull towards LeapC's joints + pose prior + temporal smoothness.
Reports point-to-mesh distance (on a dense sampling of the mesh surface), joint jitter, finger lengths.
Writes <session>/mano/: params.npz (per-frame pose, orient, transl), every --ply-every-th mesh as .ply,
and with --post SIGMA (after a fit): pose / orientation / position smoothed over time with a Gaussian of
SIGMA frames (offline, both directions), its accuracy and jitter, smoothed.npz and overlay.mp4 (fused
points grey, MANO vertices blue; front = cam_A view, side).
"""
import argparse
import csv
import json
import os
import time

import cv2
import numpy as np
import smplx
import torch
from scipy.spatial import cKDTree

from depth import hand_joints, leap_to_unit
from stereo import Cam, Unit

torch.set_num_threads(max(1, os.cpu_count() - 2))
TIPS = [745, 317, 444, 556, 673]  # MANO vertex at the tip of thumb, index, middle, ring, pinky
# MANO joints 0..15 (+ 5 tips as 16..20) -> LeapC joint index (j00..j24 in tracking.csv; 25 = wrist)
MANO_TO_LEAP = [25, 6, 7, 8, 11, 12, 13, 21, 22, 23, 16, 17, 18, 1, 2, 3, 4, 9, 14, 19, 24]
FINGERS = {"thumb": [13, 14, 15, 16], "index": [1, 2, 3, 17], "middle": [4, 5, 6, 18], "ring": [10, 11, 12, 19],
           "pinky": [7, 8, 9, 20]}  # MCP (thumb: CMC) ... tip, in the 21-joint order


class Hand:
    def __init__(self, side):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mano", f"MANO_{side.upper()}_np.pkl")
        self.m = smplx.MANO(path, is_rhand=side == "right", use_pca=False, flat_hand_mean=True)
        self.faces = self.m.faces.astype(np.int64)

    def __call__(self, betas, orient, pose, transl):
        o = self.m(betas=betas, global_orient=orient, hand_pose=pose, transl=torch.zeros_like(transl))
        v = o.vertices * 1000 + transl[:, None]
        j = torch.cat([o.joints * 1000 + transl[:, None], v[:, TIPS]], 1)
        return v, j  # (n, 778, 3), (n, 21, 3) mm


def vertex_normals(v, faces):
    """(n, 778, 3) unit normals."""
    f = torch.as_tensor(faces)
    a, b, c = v[:, f[:, 0]], v[:, f[:, 1]], v[:, f[:, 2]]
    fn = torch.cross(b - a, c - a, dim=-1)
    n = torch.zeros_like(v)
    for k in range(3):
        n.index_add_(1, f[:, k], fn)
    return n / n.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def surface_samples(v, faces, per_face=10, seed=0):
    """Dense points on the mesh surface (numpy), for point-to-mesh distances."""
    rng = np.random.default_rng(seed)
    r = rng.random((len(faces), per_face, 2))
    flip = r.sum(-1) > 1
    r[flip] = 1 - r[flip]
    a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    return (a[:, None] + r[..., :1] * (b - a)[:, None] + r[..., 1:] * (c - a)[:, None]).reshape(-1, 3)


def data_loss(v, n, pts, trim=0.8):
    """Trimmed point-to-plane distance from points (m, 3) to the nearest vertex of each mesh (one frame)."""
    d = torch.cdist(pts, v)
    j = d.argmin(1)
    r = ((pts - v[j]) * n[j]).sum(-1).abs() + 0.1 * (pts - v[j]).norm(dim=-1)  # keep it from sliding sideways
    k = max(1, int(trim * len(r)))
    return torch.topk(r, k, largest=False).values.mean()


def load_fused(session):
    z = np.load(os.path.join(session, "fused", "hands.npz"))
    P, O, F = z["points"], z["offsets"], z["frames"]
    return [(int(F[i, 0]), P[O[i]:O[i + 1]]) for i in range(len(F))]


def leap_joints_camA(session):
    """frame_id -> (21, 3) LeapC joints in MANO order, cam_A unit frame; and the hand side (majority)."""
    meta = json.load(open(os.path.join(session, "session.json")))
    cam = Cam(session, "cam_A", 800, 1.5)
    unit = Unit(meta["cameras"]["cam_A"])
    out, sides = {}, []
    with open(os.path.join(cam.dir, "tracking.csv")) as f:
        for r in csv.DictReader(f):
            fid = int(r["frame_id"])
            if fid in out:
                continue
            J, _, wrist = hand_joints(r)
            P = leap_to_unit(np.vstack([J, wrist]), unit, cam.baseline)
            out[fid] = P[MANO_TO_LEAP]
            sides.append(r["type"])
    return out, (max(set(sides), key=sides.count) if sides else None)


def sub(P, n, rng):
    return P if len(P) <= n else P[rng.choice(len(P), n, replace=False)]


def finger_lengths(j):
    """MCP (thumb: CMC) to tip along the joints, mm."""
    return {k: float(sum(np.linalg.norm(j[a] - j[b]) for a, b in zip(idx, idx[1:]))) for k, idx in FINGERS.items()}


def fit_frames(hand, betas, frames, leap, init=None, iters=60, w_joint=0.02, w_prior=0.5, w_smooth=0.0,
               prev=None, lr=0.02, opt_betas=False, npts=1500, seed=0):
    """Fit pose (+ optionally shared betas) to several frames at once. frames: [(fid, points)]."""
    rng = np.random.default_rng(seed)
    n = len(frames)
    if init is None:
        orient, pose, transl = torch.zeros(n, 3), torch.zeros(n, 45), torch.zeros(n, 3)
    else:
        orient, pose, transl = (x.clone() for x in init)
    params = [orient.requires_grad_(), pose.requires_grad_(), transl.requires_grad_()]
    if opt_betas:
        params.append(betas.requires_grad_())
    opt = torch.optim.Adam(params, lr=lr)
    pts = [torch.as_tensor(sub(p, npts, rng), dtype=torch.float32) for _, p in frames]
    lj = [torch.as_tensor(leap[f], dtype=torch.float32) if f in leap else None for f, _ in frames]
    for it in range(iters):
        opt.zero_grad()
        v, j = hand(betas.expand(n, -1), orient, pose, transl)
        nv = vertex_normals(v, hand.faces)
        loss = sum(data_loss(v[i], nv[i], pts[i]) for i in range(n)) / n
        js = [((j[i] - lj[i]).norm(dim=-1)).mean() for i in range(n) if lj[i] is not None]
        if js:
            loss = loss + w_joint * sum(js) / len(js)
        loss = loss + w_prior * (pose ** 2).mean()
        if opt_betas:
            loss = loss + 0.05 * (betas ** 2).mean()
        if prev is not None and w_smooth:
            loss = loss + w_smooth * ((pose - prev[1]) ** 2).mean() + w_smooth * 1e-3 * ((transl - prev[2]) ** 2).mean()
        loss.backward()
        opt.step()
    return orient.detach(), pose.detach(), transl.detach()


def init_from_leap(hand, betas, lj, iters=300):
    """Pose / orient / transl whose 21 joints match LeapC's (one frame)."""
    orient, pose = torch.zeros(1, 3, requires_grad=True), torch.zeros(1, 45, requires_grad=True)
    target = torch.as_tensor(lj, dtype=torch.float32)[None]
    transl = (target[:, 0] - hand(betas, orient, pose, torch.zeros(1, 3))[1][:, 0]).detach().requires_grad_()
    opt = torch.optim.Adam([orient, pose, transl], lr=0.05)
    for _ in range(iters):
        opt.zero_grad()
        _, j = hand(betas, orient, pose, transl)
        loss = (j - target).norm(dim=-1).mean() + 0.1 * (pose ** 2).mean()
        loss.backward()
        opt.step()
    return orient.detach(), pose.detach(), transl.detach()


def gaussian_smooth(x, sigma):
    """Smooth (n, k) over axis 0 with a Gaussian of sigma samples, edges reflected."""
    if sigma <= 0:
        return x
    r = int(3 * sigma)
    w = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    w /= w.sum()
    xp = np.pad(x, ((r, r), (0, 0)), mode="reflect")
    return np.stack([np.convolve(xp[:, k], w, mode="valid") for k in range(x.shape[1])], 1)


def render(P, V, center, scale, size=480):
    out = []
    for ax in ((0, 1), (2, 1)):
        img = np.zeros((size, size, 3), np.uint8)
        uv = (P[:, ax] - center[list(ax)]) * scale + size / 2
        ok = (uv >= 0).all(1) & (uv < size).all(1)
        img[uv[ok, 1].astype(int), uv[ok, 0].astype(int)] = (110, 110, 110)
        uv = (V[:, ax] - center[list(ax)]) * scale + size / 2
        for x, y in uv[(uv >= 0).all(1) & (uv < size).all(1)].astype(int):
            cv2.circle(img, (x, y), 2, (255, 150, 30), -1)  # MANO vertices on top
        out.append(img)
    return np.hstack(out)


def point_to_mesh(hand, betas, orient, pose, transl, pts):
    v, _ = hand(betas, orient, pose, transl)
    s = surface_samples(v[0].detach().numpy(), hand.faces)
    return float(np.median(cKDTree(s).query(pts)[0]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("session", nargs="?")
    ap.add_argument("--shape", nargs="+", help="still sessions to fit the person's shape on")
    ap.add_argument("--shape-frames", type=int, default=12, help="frames used for the shape, in total")
    ap.add_argument("--every", type=int, default=1, help="fit every Nth fused frame")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--ply-every", type=int, default=30)
    ap.add_argument("--post", type=float, help="smooth an existing fit over time with this sigma (frames) and report")
    args = ap.parse_args()
    shape_file = "mano_shape.json"

    if args.shape:
        side_votes, items = [], []
        per = max(1, args.shape_frames // len(args.shape))
        for s in args.shape:
            leap, side = leap_joints_camA(s)
            side_votes.append(side)
            fr = [f for f in load_fused(s) if f[0] in leap and len(f[1]) > 2000]
            items += [(s, fr[k], leap) for k in np.linspace(0, len(fr) - 1, per).astype(int)]
        side = max(set(side_votes), key=side_votes.count)
        hand = Hand(side)
        betas = torch.zeros(1, 10)
        inits = [init_from_leap(hand, betas, lp[f[0]]) for _, f, lp in items]
        init = tuple(torch.cat(x) for x in zip(*inits))
        leap_all = {f[0]: lp[f[0]] for _, f, lp in items}
        frames = [f for _, f, _ in items]
        t0 = time.time()
        o, p, t = fit_frames(hand, betas, frames, leap_all, init, iters=150, opt_betas=True, lr=0.02)
        o, p, t = fit_frames(hand, betas.detach(), frames, leap_all, (o, p, t), iters=100)
        d = [point_to_mesh(hand, betas, o[i:i + 1], p[i:i + 1], t[i:i + 1], f[1]) for i, f in enumerate(frames)]
        _, j = hand(betas, o[:1], p[:1], t[:1])
        fl = finger_lengths(j[0].detach().numpy())
        json.dump({"side": side, "betas": betas.detach().numpy().ravel().tolist(), "sessions": args.shape,
                   "frames": [f[0] for f in frames], "point_to_mesh_median_mm": float(np.median(d)),
                   "finger_lengths_mm": fl}, open(shape_file, "w"), indent=2)
        print(f"shape ({side} hand) from {len(frames)} frames in {time.time() - t0:.0f} s: point-to-mesh median "
              f"{np.median(d):.2f} mm per frame {np.round(d, 2).tolist()}")
        print("MANO finger lengths, MCP (thumb CMC) to tip, mm:", {k: round(v, 1) for k, v in fl.items()})
        print("wrote", shape_file)

    if args.session and args.post is not None:
        sh = json.load(open(shape_file))
        hand = Hand(sh["side"])
        betas = torch.tensor([sh["betas"]], dtype=torch.float32)
        out = os.path.join(args.session, "mano")
        z = np.load(os.path.join(out, "params.npz"))
        fused = dict(load_fused(args.session))
        fr = z["frames"]
        sm = {k: gaussian_smooth(z[k], args.post) for k in ("orient", "pose", "transl")}
        T = lambda a: torch.as_tensor(a, dtype=torch.float32)  # noqa: E731
        with torch.no_grad():
            V, J = hand(betas.expand(len(fr), -1), T(sm["orient"]), T(sm["pose"]), T(sm["transl"]))
        V, J = V.numpy(), J.numpy()
        for name, JJ in (("as fitted", z["joints"]), (f"smoothed, sigma {args.post:g} frames", J)):
            acc = np.linalg.norm(JJ[2:] - 2 * JJ[1:-1] + JJ[:-2], axis=-1)
            idx = range(0, len(fr), 5)
            if JJ is J:
                d = [float(np.median(cKDTree(surface_samples(V[i], hand.faces)).query(fused[int(fr[i])])[0])) for i in idx]
            else:
                with torch.no_grad():
                    d = [point_to_mesh(hand, betas, T(z["orient"][i:i + 1]), T(z["pose"][i:i + 1]),
                                       T(z["transl"][i:i + 1]), fused[int(fr[i])]) for i in idx]
            print(f"{name}: point-to-mesh median {np.median(d):.2f} mm; jitter median {np.median(acc):.2f}, "
                  f"p95 {np.percentile(acc, 95):.2f} mm/frame^2")
        np.savez_compressed(os.path.join(out, "smoothed.npz"), frames=fr, joints=J, betas=betas.numpy(),
                            sigma=args.post, **sm)
        allp = np.vstack([fused[int(f)] for f in fr[::10]])
        center = np.median(allp, 0)
        scale = 480 * 0.8 / max(np.percentile(np.abs(allp - center), 98, axis=0).max() * 2, 1)
        step = max(1, int(np.median(np.diff(fr))))
        vw = cv2.VideoWriter(os.path.join(out, "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 90 / step, (960, 480))
        for i, f in enumerate(fr):
            img = render(fused[int(f)], V[i], center, scale)
            cv2.putText(img, f"frame {f}  grey: fused points  blue: MANO", (8, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (220, 220, 220), 1)
            vw.write(img)
        vw.release()
        print("wrote", os.path.join(out, "overlay.mp4"), "and smoothed.npz")

    elif args.session:
        sh = json.load(open(shape_file))
        hand = Hand(sh["side"])
        betas = torch.tensor([sh["betas"]], dtype=torch.float32)
        leap, _ = leap_joints_camA(args.session)
        frames = load_fused(args.session)[::args.every][: args.max_frames or None]
        out = os.path.join(args.session, "mano")
        os.makedirs(out, exist_ok=True)
        prev, res, keep = None, [], []
        t0 = time.time()
        for i, (fid, P) in enumerate(frames):
            if len(P) < 500:
                continue
            if prev is None:
                if fid not in leap:
                    continue
                prev = init_from_leap(hand, betas, leap[fid])
                prev = fit_frames(hand, betas, [(fid, P)], leap, prev, iters=150)
            cur = fit_frames(hand, betas, [(fid, P)], leap, prev, iters=40, w_smooth=1.0, prev=prev)
            d = point_to_mesh(hand, betas, *cur, P)
            _, j = hand(betas, *cur)
            res.append((fid, d))
            keep.append((fid, *(x.numpy().ravel() for x in cur), j[0].numpy()))
            prev = cur
            if args.ply_every and len(keep) % args.ply_every == 1:
                v, _ = hand(betas, *cur)
                with open(os.path.join(out, f"{fid}.ply"), "w") as fh:
                    vv = v[0].numpy()
                    fh.write(f"ply\nformat ascii 1.0\nelement vertex {len(vv)}\nproperty float x\nproperty float y\n"
                             f"property float z\nelement face {len(hand.faces)}\nproperty list uchar int vertex_indices\n"
                             "end_header\n")
                    np.savetxt(fh, vv * [1, -1, -1], fmt="%.2f")
                    np.savetxt(fh, np.hstack([np.full((len(hand.faces), 1), 3), hand.faces]), fmt="%d")
            if len(res) % 50 == 0:
                print(f"  {len(res)} frames, {time.time() - t0:.0f} s, point-to-mesh median so far "
                      f"{np.median([r[1] for r in res]):.2f} mm", flush=True)
        J = np.stack([k[4] for k in keep])
        acc = np.linalg.norm(J[2:] - 2 * J[1:-1] + J[:-2], axis=-1)  # mm per frame^2
        np.savez_compressed(os.path.join(out, "params.npz"), frames=np.array([k[0] for k in keep]),
                            orient=np.stack([k[1] for k in keep]), pose=np.stack([k[2] for k in keep]),
                            transl=np.stack([k[3] for k in keep]), joints=J, betas=betas.numpy())
        d = np.array([r[1] for r in res])
        print(f"\n{len(res)} frames in {time.time() - t0:.0f} s: point-to-mesh median {np.median(d):.2f} mm "
              f"(p90 over frames {np.percentile(d, 90):.2f}); joint jitter (acceleration) median "
              f"{np.median(acc):.2f} mm/frame^2, p95 {np.percentile(acc, 95):.2f}")
        print("wrote", out)
