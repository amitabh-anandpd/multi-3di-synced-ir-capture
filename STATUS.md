# Status

## Works
- `leapc.py`: ctypes LeapC access, multi-device, no compiled bindings (SDK 6.2.0).
- `spike_two_cams.py`: both 3Dis stream IR images at the same time.

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
- 7 single-frame ID gaps per 10 s show up even with one camera alone, so it isn't USB load. Look into it in M4.

## Log
- 2026-10-26: M1 passed. Root cause of the earlier zero-frame result: devices have to be opened
  from device events inside the poll loop (subscribe, then set the policy). Enumerating with
  LeapGetDeviceList after polling gave no frames. No Control Panel "background apps" setting is needed.
