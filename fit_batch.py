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


def refine(ob, x, iters, lr, passes, pass_iters):
    """Optimise a run of frames, then propagation, like fit_mano.py's frame-by-frame tracking: every frame also tries
    the result of the frame 1, 2, 4, 8 ... before / after it and keeps whichever fits its own points better."""
    n = len(x[0])
    x = ob.optimise(x, iters, lr)
    for sh in SHIFTS[:passes]:
        if abs(sh) >= n:
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
    return x


def switch_steps(orient, frames, max_gap=6):
    """Steps i (between frame i and i + 1) where the hand turns > 90 deg within max_gap camera frames: impossible for
    a real hand (< 0.1 s), so a switch between the palm-front and palm-back fits of a flat hand seen edge-on."""
    from scipy.spatial.transform import Rotation
    r = Rotation.from_rotvec(orient.cpu().numpy())
    turn = np.degrees((r[:-1].inv() * r[1:]).magnitude())
    return np.nonzero((turn > 90) & (np.diff(frames) <= max_gap))[0]


def repair(setup, x, frames, lr=0.02, passes=8, pass_iters=40, iters=150, chunk=256):
    """Make the orientation consistent through the session.

    Cut at the switches into stretches; refit every stretch starting from its neighbours' orientation and finger pose
    (= the other one of the two fits); then of the two ways to pick per stretch that have no switch at all (current,
    other, current, ... or other, current, other, ...), keep the one with the lower total cost (points + LeapC + prior).

    EXPERIMENTAL, off by default (2026-11-12 test, 2026-11-11_232756 frames 0-130): surface 2.77 -> 2.54 mm, but a new
    switch appeared inside the refitted stretch and its joints moved from 18 to 37 mm from LeapC's. The hand really
    turns there, and LeapC agrees with the fit on both sides of the original switch, so forcing continuity made the
    joint labels worse. Deciding which side is right needs ground truth (PLAN.md P2, hand-clicked keypoints)."""
    hand, betas, faces, pts, lj, has = setup
    steps = switch_steps(x[0], frames)
    if not len(steps):
        return x, "no switches"
    n = len(frames)
    bounds = [0] + [int(i) + 1 for i in steps] + [n]
    spans = list(zip(bounds[:-1], bounds[1:]))
    cur_cost, alt, alt_cost = [], [], []
    for k, (a, b) in enumerate(spans):
        idx = torch.arange(a, b, device=DEV)
        src = []
        if k > 0:
            src.append(a - 1)
        if k < len(spans) - 1:
            src.append(b)
        src = torch.tensor(src, device=DEV)
        near = src[(idx[:, None] - src[None]).abs().argmin(1)]  # nearest frame of a neighbouring stretch
        init = [x[0][near], x[1][near], x[2][a:b]]
        res, c0, c1 = [], 0.0, 0.0
        for s in range(a, b, chunk):
            e = min(b, s + chunk)
            ob = Objective(hand, betas, faces, pts[s:e], lj[s:e], has[s:e].float())
            y = refine(ob, [v[s - a:e - a] for v in init], iters, lr, passes, pass_iters)
            with torch.no_grad():
                c0 += float(ob.per_frame(*(v[s:e] for v in x)).sum())
                c1 += float(ob.per_frame(*y).sum())
            res.append(y)
        alt.append([torch.cat(v) for v in zip(*res)])
        cur_cost.append(c0)
        alt_cost.append(c1)
    pick = []
    for first in (0, 1):  # 0: stretch 0 keeps its fit, 1: stretch 0 takes the other
        use = [(k + first) % 2 for k in range(len(spans))]
        pick.append((sum(alt_cost[k] if u else cur_cost[k] for k, u in enumerate(use)), use))
    total, use = min(pick)
    out = [v.clone() for v in x]
    for k, ((a, b), u) in enumerate(zip(spans, use)):
        if u:
            for v, w in zip(out, alt[k]):
                v[a:b] = w
    left = len(switch_steps(out[0], frames))
    msg = (f"{len(steps)} switches, {len(spans)} stretches; per-frame cost kept {pick[0][0] / n:.3f} vs other "
           f"{pick[1][0] / n:.3f} -> option {pick.index((total, use))}, other fit used for "
           f"{sum(b - a for (a, b), u in zip(spans, use) if u)} of {n} frames; switches left: {left}")
    return out, msg


def load_setup(z, first=None):
    hand = Hand(str(z["side"]))
    hand.m.to(DEV)
    faces = torch.as_tensor(hand.faces, device=DEV)
    betas = torch.as_tensor(z["betas"], device=DEV)[None]
    T = lambda a: torch.as_tensor(a, dtype=torch.float32, device=DEV)  # noqa: E731
    pts, lj, has = T(z["points"]), T(z["leap"]), torch.as_tensor(z["has_leap"], device=DEV)
    if first:
        pts, lj, has = pts[first[0]:first[1]], lj[first[0]:first[1]], has[first[0]:first[1]]
    return hand, betas, faces, pts, lj, has


def save(path, frames, setup, x):
    hand, betas = setup[0], setup[1]
    with torch.no_grad():
        _, j = hand(betas.expand(len(frames), -1), *x)
    C = lambda v: v.cpu().numpy()  # noqa: E731
    np.savez_compressed(path, frames=frames, orient=C(x[0]), pose=C(x[1]), transl=C(x[2]), joints=C(j), betas=C(betas))


def repair_file(path):
    """--repair: fix the switches in an existing <input>_params.npz (keeps the old one as *_params_before_repair.npz)."""
    z = np.load(path)
    out = path.replace(".npz", "_params.npz")
    p = np.load(out)
    setup = load_setup(z)
    T = lambda a: torch.as_tensor(a, dtype=torch.float32, device=DEV)  # noqa: E731
    t0 = time.time()
    x, msg = repair(setup, [T(p["orient"]), T(p["pose"]), T(p["transl"])], p["frames"])
    shutil.copy(out, out.replace("_params.npz", "_params_before_repair.npz"))
    save(out, p["frames"], setup, x)
    print(f"{path}: {msg} ({time.time() - t0:.0f} s) -> {out}", flush=True)


def fit(path, iters=200, chunk=256, lr=0.02, passes=12, pass_iters=40, first=None, fix=False):
    z = np.load(path)
    setup = load_setup(z, first)
    hand, betas, faces, pts, lj, has = setup
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
        x = refine(ob, [a[s:e] for a in (o0, p0, t0_)], iters, lr, passes, pass_iters)
        keep = slice(0 if s == 0 else 8, None)
        res.append([a[keep] for a in x])
        if e == n:
            break
    x = [torch.cat(v) for v in zip(*res)]
    frames = z["frames"][first[0]:first[1]] if first else z["frames"]
    msg = ""
    if fix:
        x, msg = repair(setup, x, frames, lr=lr, pass_iters=pass_iters, chunk=chunk)
        msg = "; orientation: " + msg
    out = path.replace(".npz", "_params.npz")
    save(out, frames, setup, x)
    print(f"{path}: {n} frames on {DEV} in {time.time() - t0:.0f} s (start {t_init:.0f} s){msg} -> {out}", flush=True)


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
    g.add_argument("--repair", nargs="+", metavar="NPZ", help="fix palm/back switches in existing <NPZ>_params.npz")
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
    elif args.repair:
        for p in args.repair:
            repair_file(p)
    else:
        unpack(args.unpack)
