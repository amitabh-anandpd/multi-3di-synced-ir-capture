"""Live view of every camera's left image, for aiming the rig. Press q (in the window) to quit.

  python tools/live_view.py

Use it instead of the Ultraleap Control Panel, and close it before capturing: any other
client of the service lowers capture FPS.
"""
import ctypes
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import leapc  # noqa: E402
from list_devices import CONFIG  # noqa: E402

names = {s: n for n, s in json.load(open(CONFIG))["cameras"].items()}
conn, devices = leapc.connect()
if not devices:
    raise SystemExit("No cameras found.")
label = {d["id"]: names.get(d["serial"], d["serial"][-6:]) for d in devices}
latest = {}
msg = leapc.LEAP_CONNECTION_MESSAGE()
try:
    while True:
        for _ in range(20):  # drain a few events per screen refresh
            if leapc.lib.LeapPollConnection(conn, 5, ctypes.byref(msg)) != leapc.RS_SUCCESS:
                break
            if msg.type == leapc.EVT_IMAGE and msg.device_id in label:
                img = leapc.image_event(msg).image[0]
                p = img.properties
                latest[msg.device_id] = np.frombuffer(leapc.image_bytes(img), np.uint8).reshape(p.height, p.width)
        if latest:
            tiles = []
            for dev_id in sorted(label):
                tile = cv2.cvtColor(latest.get(dev_id, np.zeros((384, 384), np.uint8)), cv2.COLOR_GRAY2BGR)
                cv2.putText(tile, label[dev_id], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                tiles.append(tile)
            cv2.imshow("live view (q to quit)", np.hstack(tiles))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cv2.destroyAllWindows()
    leapc.close(conn, devices)
