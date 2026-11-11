"""Batched MANO fit: all frames of a session optimised together (GPU if there is one), instead of one after another.

Same loss as fit_mano.py (trimmed point-to-plane at the nearest vertex, weak pull to LeapC's joints, pose prior), but
every frame starts from LeapC's joints, neighbouring frames are tied by a smoothness term, and propagation passes let
each frame take over the result of the frame 1, 2, 4, 8, 16 before / after it when that fits its points better (in
place of fit_mano.py's frame-by-frame tracking). Without the passes, a stretch of frames stays stuck near LeapC's
start (2026-11-06_035901 frames 300-380: 2.93 mm); with them 2.71 mm vs fit_mano.py's 2.79, and 5.5 vs 9.6 mm from
LeapC's joints (mean offset removed). On the CPU it is slower than fit_mano.py (~7 s per frame); use a GPU.

  1. locally (needs the session's tracking, rig.json, units/):
       python fit_batch.py --pack sessions/2026-11-11_223221 sessions/2026-11-11_223307
     -> fit_inputs/<session>.npz (~15 MB each: points subsampled per frame, LeapC joints, the person's MANO shape)
  2. anywhere with torch + smplx + mano/*_np.pkl (Kaggle GPU: fit_kaggle.ipynb; or here on the CPU):
       python fit_batch.py --fit fit_inputs/2026-11-11_223221.npz
     -> fit_inputs/<session>_params.npz (same layout as fit_mano.py's params.npz)
  3. locally: python fit_batch.py --unpack fit_inputs   (copies each *_params.npz to <session>/mano/params.npz)
     then    python fit_mano.py sessions/2026-11-11_223221 --post 1   (smoothing, accuracy, overlay.mp4)
"""
import argparse
import glob
import json
import os
import shutil
import time

import numpy as np
import torch

from fit_mano import Hand

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SHIFTS = [1, -1, 2, -2, 4, -4, 8, -8, 16, -16, 1, -1]


def pack(session, out_dir, every=1, npts=1500, min_pts=500, seed=0):
    from fit_mano import leap_joints_camA, load_fused
    sh = json.load(open("mano_shape.json"))
    leap, _ = leap_joints_camA(session)
    rng = np.random.default_rng(seed)
    F, P, L, H = [], [], [], []
    for fid, p in load_fused(session)[::every]:
        if len(p) < min_pts:
            continue
        F.append(fid)
        P.append(p[rng.choice(len(p), npts, replace=len(p) < npts)])
        H.append(fid in leap)
        L.append(leap[fid] if fid in leap else np.zeros((21, 3)))
    name = os.path.basename(os.path.normpath(session))
    path = os.path.join(out_dir, f"{name}.npz")
    np.savez_compressed(path, session=name, frames=np.array(F), points=np.array(P, np.float32),
                        leap=np.array(L, np.float32), has_leap=np.array(H), side=sh["side"],
                        betas=np.array(sh["betas"], np.float32))
    print(f"{session}: {len(F)} frames ({sum(H)} with LeapC joints) -> {path} ({os.path.getsize(path) / 1e6:.0f} MB)")


def normals(v, faces):
    a, b, c = v[:, faces[:, 0]], v[:, faces[:, 1]], v[:, faces[:, 2]]
    fn = torch.cross(b - a, c - a, dim=-1)
    n = torch.zeros_like(v)
    for k in range(3):
        n.index_add_(1, faces[:, k], fn)
    return n / n.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def data_loss(v, n, pts, trim=0.8):
    """Batched fit_mano.data_loss, per frame (B,): (B, 778, 3) mesh, normals; (B, M, 3) points."""
    with torch.no_grad():
        j = torch.cdist(pts, v).argmin(-1)
    idx = j[..., None].expand(-1, -1, 3)
    vj, nj = torch.gather(v, 1, idx), torch.gather(n, 1, idx)
    r = ((pts - vj) * nj).sum(-1).abs() + 0.1 * (pts - vj).norm(dim=-1)
    return torch.topk(r, max(1, int(trim * pts.shape[1])), dim=1, largest=False).values.mean(1)


class Objective:
    """fit_mano.fit_frames' loss for a run of consecutive frames; per frame (for choosing) or total (to optimise)."""

    def __init__(self, hand, betas, faces, P, J, Hm, w_joint=0.02, w_prior=0.5, w_smooth=1.0, trim=0.8):
        self.__dict__.update(locals())

    def per_frame(self, orient, pose, transl):
        B = len(pose)
        v, j = self.hand(self.betas.expand(B, -1), orient, pose, transl)
        f = data_loss(v, normals(v, self.faces), self.P, self.trim) + self.w_prior * (pose ** 2).mean(1)
        return f + self.w_joint * (j - self.J).norm(dim=-1).mean(1) * self.Hm

    def total(self, orient, pose, transl):
        loss = self.per_frame(orient, pose, transl).mean()
        if len(pose) > 1:
            loss = loss + self.w_smooth * ((pose[1:] - pose[:-1]) ** 2).mean() \
                + self.w_smooth * 1e-3 * ((transl[1:] - transl[:-1]) ** 2).mean()
        return loss

    def optimise(self, x, iters, lr):
        x = [a.clone().requires_grad_() for a in x]
        opt = torch.optim.Adam(x, lr=lr)
        for _ in range(iters):
            opt.zero_grad()
            self.total(*x).backward()
            opt.step()
        return [a.detach() for a in x]


def init_from_leap(hand, betas, lj, iters=300):
    """Batched fit_mano.init_from_leap: (B, 21, 3) joints -> orient, pose, transl."""
    B = len(lj)
    orient = torch.zeros(B, 3, device=DEV, requires_grad=True)
    pose = torch.zeros(B, 45, device=DEV, requires_grad=True)
    z = torch.zeros(B, 3, device=DEV)
    transl = (lj[:, 0] - hand(betas.expand(B, -1), orient, pose, z)[1][:, 0]).detach().requires_grad_()
    opt = torch.optim.Adam([orient, pose, transl], lr=0.05)
    for _ in range(iters):
        opt.zero_grad()
        _, j = hand(betas.expand(B, -1), orient, pose, transl)
        loss = (j - lj).norm(dim=-1).mean() + 0.1 * (pose ** 2).mean()
        loss.backward()
        opt.step()
    return orient.detach(), pose.detach(), transl.detach()


def fit(path, iters=200, chunk=256, lr=0.02, passes=12, pass_iters=40, first=None):
    z = np.load(path)
    hand = Hand(str(z["side"]))
    hand.m.to(DEV)
    faces = torch.as_tensor(hand.faces, device=DEV)
    betas = torch.as_tensor(z["betas"], device=DEV)[None]
    T = lambda a: torch.as_tensor(a, dtype=torch.float32, device=DEV)  # noqa: E731
    pts, lj, has = T(z["points"]), T(z["leap"]), torch.as_tensor(z["has_leap"], device=DEV)
    if first:
        pts, lj, has = pts[first[0]:first[1]], lj[first[0]:first[1]], has[first[0]:first[1]]
    n = len(pts)
    if not has.any():
        raise SystemExit(f"{path}: no frame has LeapC joints to start from")
    t0 = time.time()
    # start: LeapC's joints; frames without them take the nearest frame that has them
    o0, p0, t0_ = (torch.zeros(n, k, device=DEV) for k in (3, 45, 3))
    hi = torch.nonzero(has).ravel()
    for s in range(0, len(hi), 1024):
        i = hi[s:s + 1024]
        o0[i], p0[i], t0_[i] = init_from_leap(hand, betas, lj[i])
    near = hi[(torch.arange(n, device=DEV)[:, None] - hi[None]).abs().argmin(1)]
    o0, p0, t0_ = o0[near], p0[near], t0_[near]
    t_init = time.time() - t0
    # joint fit in chunks of consecutive frames, neighbours tied together
    res, step = [], chunk - 8  # chunks overlap by 8 frames; the overlap's second copy is dropped
    for s in range(0, n, step):
        e = min(n, s + chunk)
        ob = Objective(hand, betas, faces, pts[s:e], lj[s:e], has[s:e].float())
        x = ob.optimise([a[s:e] for a in (o0, p0, t0_)], iters, lr)
        # propagation, like fit_mano.py's frame-by-frame tracking: every frame also tries the result of the frame
        # 1, 2, 4, 8 ... before / after it and keeps whichever fits its own points better
        for sh in SHIFTS[:passes]:
            if abs(sh) >= e - s:
                continue
            cand = ob.optimise([torch.roll(a, sh, 0) for a in x], pass_iters, lr)
            with torch.no_grad():
                better = ob.per_frame(*cand) < ob.per_frame(*x)
                if sh > 0:
                    better[:sh] = False  # rolled around from the end
                else:
                    better[sh:] = False
            x = [torch.where(better[:, None], c, a) for c, a in zip(cand, x)]
            x = ob.optimise(x, pass_iters, lr)
        keep = slice(0 if s == 0 else 8, None)
        res.append([a[keep] for a in x])
        if e == n:
            break
    orient, pose, transl = (torch.cat(x) for x in zip(*res))
    with torch.no_grad():
        _, j = hand(betas.expand(n, -1), orient, pose, transl)
    out = path.replace(".npz", "_params.npz")
    C = lambda x: x.cpu().numpy()  # noqa: E731
    frames = z["frames"][first[0]:first[1]] if first else z["frames"]
    np.savez_compressed(out, frames=frames, orient=C(orient), pose=C(pose), transl=C(transl), joints=C(j),
                        betas=C(betas))
    print(f"{path}: {n} frames on {DEV} in {time.time() - t0:.0f} s (start {t_init:.0f} s) -> {out}", flush=True)


def unpack(folder):
    for p in sorted(glob.glob(os.path.join(folder, "*_params.npz"))):
        s = os.path.join("sessions", os.path.basename(p)[:-len("_params.npz")])
        os.makedirs(os.path.join(s, "mano"), exist_ok=True)
        shutil.copy(p, os.path.join(s, "mano", "params.npz"))
        print(f"{p} -> {s}/mano/params.npz;  next: python fit_mano.py {s} --post 1")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pack", nargs="+", metavar="SESSION")
    g.add_argument("--fit", nargs="+", metavar="NPZ")
    g.add_argument("--unpack", metavar="FOLDER")
    ap.add_argument("--out", default="fit_inputs")
    ap.add_argument("--every", type=int, default=1, help="--pack: every Nth fused frame")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--chunk", type=int, default=256, help="frames optimised together")
    ap.add_argument("--passes", type=int, default=12, help="propagation passes (0 = none)")
    ap.add_argument("--range", type=int, nargs=2, help="--fit only frames [a, b) of the packed list (testing)")
    args = ap.parse_args()
    if args.pack:
        os.makedirs(args.out, exist_ok=True)
        for s in args.pack:
            pack(s, args.out, args.every)
    elif args.fit:
        for p in args.fit:
            fit(p, args.iters, args.chunk, passes=args.passes, first=args.range)
    else:
        unpack(args.unpack)
