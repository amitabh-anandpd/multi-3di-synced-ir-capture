"""Minimal ctypes mirror of LeapC (checked against LeapC.h, SDK 6.2.0).

Only covers what capture needs. Loads LeapC.dll from the Ultraleap install,
so no compiled bindings are required. Override the path with LEAPC_DLL.
"""
import ctypes
import os
import time
from ctypes import POINTER, byref, c_char_p, c_float, c_int32, c_int64, c_uint32, c_uint64, c_void_p, sizeof

import numpy as np

DLL_PATH = os.environ.get("LEAPC_DLL", r"C:\Program Files\Ultraleap\LeapSDK\lib\x64\LeapC.dll")

# eLeapRS
RS_SUCCESS = 0x00000000
RS_INSUFFICIENT_BUFFER = 0xE2010003
RS_TIMEOUT = 0xE2010004

CONNECTION_CONFIG_MULTI_DEVICE_AWARE = 0x00000001
POLICY_IMAGES = 0x00000002

# eLeapEventType
EVT_CONNECTION = 1
EVT_DEVICE = 3
EVT_DEVICE_FAILURE = 4
EVT_POLICY = 5
EVT_DEVICE_LOST = 0x104
EVT_DROPPED_FRAME = 0x108
EVT_IMAGE = 0x109

DISTORTION_N = 64


class _Packed(ctypes.Structure):
    _pack_ = 1  # LeapC.h uses #pragma pack(1)


class LEAP_DEVICE_REF(_Packed):
    _fields_ = [("handle", c_void_p), ("id", c_uint32)]


class LEAP_CONNECTION_CONFIG(_Packed):
    _fields_ = [("size", c_uint32), ("flags", c_uint32), ("server_namespace", c_char_p), ("tracking_origin", c_int32)]


class LEAP_DEVICE_EVENT(_Packed):
    _fields_ = [("flags", c_uint32), ("device", LEAP_DEVICE_REF), ("status", c_uint32)]


class LEAP_DEVICE_INFO(_Packed):
    _fields_ = [("size", c_uint32), ("status", c_uint32), ("caps", c_uint32), ("pid", c_int32),
                ("baseline", c_uint32), ("serial_length", c_uint32), ("serial", c_char_p),
                ("h_fov", c_float), ("v_fov", c_float), ("range", c_uint32)]


class LEAP_FRAME_HEADER(_Packed):
    _fields_ = [("reserved", c_void_p), ("frame_id", c_int64), ("timestamp", c_int64)]


class LEAP_IMAGE_PROPERTIES(_Packed):
    _fields_ = [("type", c_int32), ("format", c_int32), ("bpp", c_uint32), ("width", c_uint32), ("height", c_uint32),
                ("x_scale", c_float), ("y_scale", c_float), ("x_offset", c_float), ("y_offset", c_float)]


class LEAP_IMAGE(_Packed):
    _fields_ = [("properties", LEAP_IMAGE_PROPERTIES), ("matrix_version", c_uint64),
                ("distortion_matrix", c_void_p),  # LEAP_DISTORTION_MATRIX*: 64x64 (x, y) floats
                ("data", c_void_p), ("offset", c_uint32)]


class LEAP_IMAGE_EVENT(_Packed):
    _fields_ = [("info", LEAP_FRAME_HEADER), ("image", LEAP_IMAGE * 2), ("calib", c_void_p)]


class LEAP_VECTOR(_Packed):
    _fields_ = [("x", c_float), ("y", c_float), ("z", c_float)]


class LEAP_CONNECTION_MESSAGE(_Packed):
    _fields_ = [("size", c_uint32), ("type", c_int32), ("pointer", c_void_p), ("device_id", c_uint32)]


lib = ctypes.CDLL(DLL_PATH)


def _sig(name, res, *args):
    f = getattr(lib, name)
    f.restype, f.argtypes = res, list(args)


_sig("LeapGetNow", c_int64)
_sig("LeapCreateConnection", c_uint32, POINTER(LEAP_CONNECTION_CONFIG), POINTER(c_void_p))
_sig("LeapOpenConnection", c_uint32, c_void_p)
_sig("LeapPollConnection", c_uint32, c_void_p, c_uint32, POINTER(LEAP_CONNECTION_MESSAGE))
_sig("LeapOpenDevice", c_uint32, LEAP_DEVICE_REF, POINTER(c_void_p))
_sig("LeapGetDeviceInfo", c_uint32, c_void_p, POINTER(LEAP_DEVICE_INFO))
_sig("LeapSubscribeEvents", c_uint32, c_void_p, c_void_p)
_sig("LeapSetPolicyFlagsEx", c_uint32, c_void_p, c_void_p, c_uint64, c_uint64)
_sig("LeapCameraMatrixEx", None, c_void_p, c_void_p, c_int32, POINTER(c_float))
_sig("LeapDistortionCoeffsEx", None, c_void_p, c_void_p, c_int32, POINTER(c_float))
_sig("LeapExtrinsicCameraMatrixEx", None, c_void_p, c_void_p, c_int32, POINTER(c_float))
_sig("LeapRectilinearToPixelEx", LEAP_VECTOR, c_void_p, c_void_p, c_int32, LEAP_VECTOR)
_sig("LeapCloseDevice", None, c_void_p)
_sig("LeapCloseConnection", None, c_void_p)
_sig("LeapDestroyConnection", None, c_void_p)


def check(rs, what=""):
    if rs != RS_SUCCESS:
        raise RuntimeError(f"LeapC {what} failed: 0x{rs:08X}")


def _serial(h):
    info = LEAP_DEVICE_INFO(size=sizeof(LEAP_DEVICE_INFO))
    rs = lib.LeapGetDeviceInfo(h, byref(info))  # first call only fills serial_length
    if rs not in (RS_SUCCESS, RS_INSUFFICIENT_BUFFER):
        check(rs, "LeapGetDeviceInfo(length)")
    buf = ctypes.create_string_buffer(info.serial_length)
    info.serial = ctypes.cast(buf, c_char_p)
    check(lib.LeapGetDeviceInfo(h, byref(info)), "LeapGetDeviceInfo")
    return buf.value.decode(), info.pid, info.baseline


def connect(settle_s=1.5, timeout_s=5.0, serials=None):
    """Open a multi-device-aware connection with the images policy on each device.

    Devices must be opened from their device events, inside the poll loop:
    subscribe first, then set the policy. Opening them via LeapGetDeviceList after
    polling gives zero frames (measured, SDK 6.2.0).
    `serials` limits which devices get images.
    Returns (conn, [{'id', 'handle', 'serial', 'pid', 'baseline_um'}]).
    """
    cfg = LEAP_CONNECTION_CONFIG(sizeof(LEAP_CONNECTION_CONFIG), CONNECTION_CONFIG_MULTI_DEVICE_AWARE, None, 0)
    conn = c_void_p()
    check(lib.LeapCreateConnection(byref(cfg), byref(conn)), "LeapCreateConnection")
    check(lib.LeapOpenConnection(conn), "LeapOpenConnection")
    msg, devices = LEAP_CONNECTION_MESSAGE(), []
    deadline, connected = time.monotonic() + timeout_s, False
    while time.monotonic() < deadline:
        if lib.LeapPollConnection(conn, 100, byref(msg)) != RS_SUCCESS:
            continue
        if msg.type == EVT_CONNECTION and not connected:
            connected, deadline = True, time.monotonic() + settle_s
        elif msg.type == EVT_DEVICE:
            ref = ctypes.cast(msg.pointer, POINTER(LEAP_DEVICE_EVENT)).contents.device
            h = c_void_p()
            check(lib.LeapOpenDevice(ref, byref(h)), "LeapOpenDevice")
            serial, pid, baseline_um = _serial(h)
            if serials and serial not in serials:
                lib.LeapCloseDevice(h)
                continue
            check(lib.LeapSubscribeEvents(conn, h), "LeapSubscribeEvents")
            check(lib.LeapSetPolicyFlagsEx(conn, h, POLICY_IMAGES, 0), "LeapSetPolicyFlagsEx")
            devices.append({"id": ref.id, "handle": h, "serial": serial, "pid": pid, "baseline_um": baseline_um})
    if not connected:
        raise RuntimeError("No connection to the Ultraleap Tracking Service. Is it running?")
    return conn, devices


def image_event(msg):
    return ctypes.cast(msg.pointer, POINTER(LEAP_IMAGE_EVENT)).contents


def image_bytes(img):
    """Copy one IR image out of LeapC's buffer (valid only until the next poll)."""
    p = img.properties
    return ctypes.string_at(img.data + img.offset, p.width * p.height * p.bpp)


def distortion(img):
    """Copy the 64x64 (x, y) float32 distortion map of one image.

    Layout (verified against LeapRectilinearToPixelEx, 0.6 px median error):
    m[r, c] is the raw pixel for ray slope (rx, ry) = (c/63*8 - 4, 4 - r/63*8),
    as normalised (x, y); pixel = (x*W - 0.5, y*H + 0.5). Rows run bottom-up.
    """
    return np.frombuffer(ctypes.string_at(img.distortion_matrix, DISTORTION_N * DISTORTION_N * 2 * 4),
                         np.float32).reshape(DISTORTION_N, DISTORTION_N, 2)


def calibration(conn, h):
    """Factory calibration of both imagers. Only valid after the device has sent an image.

    K: OpenCV camera matrix (3x3). D: 8 OpenCV distortion coeffs.
    extrinsic: 4x4 (LeapC returns column-major; transposed here to row-major).
    """
    out = {}
    for name, persp in (("left", 1), ("right", 2)):  # eLeapPerspectiveType_stereo_left/right
        k, d, e = (c_float * 9)(), (c_float * 8)(), (c_float * 16)()
        lib.LeapCameraMatrixEx(conn, h, persp, k)
        lib.LeapDistortionCoeffsEx(conn, h, persp, d)
        lib.LeapExtrinsicCameraMatrixEx(conn, h, persp, e)
        out[name] = {"K": np.array(k).reshape(3, 3).tolist(), "D": list(d),
                     "extrinsic": np.array(e).reshape(4, 4).T.tolist()}
    return out


def close(conn, devices):
    for d in devices:
        lib.LeapCloseDevice(d["handle"])
    lib.LeapCloseConnection(conn)
    lib.LeapDestroyConnection(conn)
