# Status

## Works
- `leapc.py`: ctypes LeapC access, multi-device, no compiled bindings (SDK 6.2.0).
- `spike_two_cams.py`: both 3Dis stream IR images at the same time.
- `list_devices.py`: stable names (cam_A = …6195, cam_B = …6230). Survived swapping the USB ports.
- `capture_multi.py`: simultaneous capture of both cameras into a session folder.

## M3 capture (2026-10-26, 60 s, both cameras, one Intel USB 3.2 controller, no hub)

| camera | frames | FPS | dropped | max writer queue | latency |
|---|---|---|---|---|---|
| cam_A | 5383 | 89.7 | 0 | 3 | 8.0 ms |
| cam_B | 5383 | 89.7 | 0 | 5 | 8.0 ms |

Disk: ~0.85 GB per camera per minute (PNG level 1).

## Measured
2026-10-26, SDK 6.2.0, 2 × 3Di, 10 s runs:

| run | camera | FPS | frame_id gaps | ts jitter | service→host latency |
|---|---|---|---|---|---|
| together | …6195 (id 1) | 89.3 | 7 | 1.03 ms | 9.5 ms |
| together | …6230 (id 2) | 89.3 | 7 | 1.05 ms | 10.0 ms |
| alone | …6195 | 89.3 | 7 | 1.01 ms | 8.4 ms |
| alone | …6230 | 89.3 | 7 | 0.98 ms | 8.3 ms |

Images: 384×384 per eye, 1 byte/pixel, IR format.

## Open issues
- A bad USB connection made cam_B silent. Control Panel warning: "USB hub or low powered port".
  Service log: "Broken device detected, missing frame data". Fixed by reseating the cable and restarting.
  capture_multi.py now aborts if a camera sends nothing in 2 s.
- Disk use (~1.7 GB/min for 2 cams) limits long sessions. Fine for bursts.

## Resolved
- 7 single-frame ID gaps per 10 s in M1: gone in the M3 run (0 drops). Probably the same failing USB link.

## Log
- 2026-10-26: M1 passed. Root cause of the earlier zero-frame result: devices have to be opened
  from device events inside the poll loop (subscribe, then set the policy). Enumerating with
  LeapGetDeviceList after polling gave no frames. No Control Panel "background apps" setting is needed.
- 2026-10-26: M2 passed. Names stayed stable after unplugging and swapping the ports.
- 2026-10-26: M3 passed after cam_B's USB fault was fixed (reseat + restart). 60 s, 0 drops on both cameras.
