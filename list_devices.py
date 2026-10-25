"""List connected 3Dis and give each serial a stable name in config.json.

Existing serial -> name mappings are never changed; new serials get the next free cam_X.
"""
import json
import os
import string

import leapc

CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def assign_names(cameras, serials):
    """cameras: {name: serial}. Adds unseen serials under the next free cam_X name."""
    known = set(cameras.values())
    free = (f"cam_{c}" for c in string.ascii_uppercase if f"cam_{c}" not in cameras)
    for s in sorted(serials):  # sorted so first-time naming doesn't depend on plug-in order
        if s not in known:
            cameras[next(free)] = s
    return cameras


if __name__ == "__main__":
    assert assign_names({"cam_A": "y"}, ["x", "y", "z"]) == {"cam_A": "y", "cam_B": "x", "cam_C": "z"}

    conn, devices = leapc.connect()
    leapc.close(conn, devices)
    cfg = json.load(open(CONFIG)) if os.path.exists(CONFIG) else {"cameras": {}}
    assign_names(cfg["cameras"], [d["serial"] for d in devices])
    with open(CONFIG, "w") as f:
        json.dump(cfg, f, indent=2)

    connected = {d["serial"] for d in devices}
    for name, serial in sorted(cfg["cameras"].items()):
        print(f"{name}  {serial}  {'connected' if serial in connected else 'not connected'}")
