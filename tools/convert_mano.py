"""N7: convert the official MANO .pkl files to plain numpy, so they load without chumpy.

  python tools/convert_mano.py            (mano/MANO_LEFT.pkl, MANO_RIGHT.pkl -> mano/MANO_*_np.pkl)

The MANO download stores some arrays as chumpy objects; chumpy doesn't install on Python 3.11 with
current numpy. Unpickling with a stand-in class keeps each object's state, and its array ('x') is
taken out. The licence doesn't allow sharing these files: mano/ is git-ignored.
"""
import os
import pickle
import sys

import numpy as np
import scipy.sparse

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mano")


class _Stub:
    def __init__(self, *a, **k):
        self.state = {}

    def __setstate__(self, state):
        self.state = state


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return _Stub
        return super().find_class(module, name)


def plain(v):
    if isinstance(v, _Stub):
        st = v.state if isinstance(v.state, dict) else {}
        if "x" not in st and "a" in st and "idxs" in st:  # chumpy Select: elements idxs of another object
            sel = np.asarray(plain(st["a"])).ravel()[np.asarray(st["idxs"]).ravel()]
            shape = st.get("preferred_shape")
            return sel.reshape(shape) if shape is not None else sel
        x = st.get("x")
        if x is None:
            raise ValueError(f"chumpy object without 'x': {list(v.state) if isinstance(v.state, dict) else type(v.state)}")
        return np.asarray(x)
    return v


for side in ("LEFT", "RIGHT"):
    src, dst = os.path.join(ROOT, f"MANO_{side}.pkl"), os.path.join(ROOT, f"MANO_{side}_np.pkl")
    if not os.path.exists(src):
        sys.exit(f"missing {src}")
    with open(src, "rb") as f:
        data = _Unpickler(f, encoding="latin1").load()
    out = {k: plain(v) for k, v in data.items()}
    for k, v in out.items():
        kind = type(v).__name__
        shape = getattr(v, "shape", "")
        if not isinstance(v, (np.ndarray, scipy.sparse.spmatrix, str, list, dict, int, float)):
            print(f"  note: {k} is {kind}")
        print(f"{side} {k}: {kind} {shape}")
    with open(dst, "wb") as f:
        pickle.dump(out, f)
    print("wrote", dst)
