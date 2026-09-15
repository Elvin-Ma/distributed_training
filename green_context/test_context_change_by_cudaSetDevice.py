#!/usr/bin/env python3
"""Show whether cudaSetDevice(0) creates or selects a new driver context."""

import ctypes

import torch


DRIVER = ctypes.CDLL("libcuda.so.1")
RUNTIME = ctypes.CDLL("libcudart.so")

DRIVER.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
DRIVER.cuCtxGetCurrent.restype = ctypes.c_int
DRIVER.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
DRIVER.cuCtxSetCurrent.restype = ctypes.c_int
RUNTIME.cudaGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
RUNTIME.cudaGetDevice.restype = ctypes.c_int
RUNTIME.cudaSetDevice.argtypes = [ctypes.c_int]
RUNTIME.cudaSetDevice.restype = ctypes.c_int


def check(label: str, code: int) -> None:
    if code != 0:
        raise RuntimeError(f"{label} failed with CUDA error {code}")


def state(label: str) -> int:
    context = ctypes.c_void_p()
    device = ctypes.c_int(-1)
    check("cuCtxGetCurrent", DRIVER.cuCtxGetCurrent(ctypes.byref(context)))
    check("cudaGetDevice", RUNTIME.cudaGetDevice(ctypes.byref(device)))
    value = context.value or 0
    print(f"{label}: device={device.value} context=0x{value:x}", flush=True)
    return value


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.cuda.set_device(0)
    check("cudaSetDevice(primary)", RUNTIME.cudaSetDevice(0))
    primary = state("primary before")

    check("cudaSetDevice(primary, same device)", RUNTIME.cudaSetDevice(0))
    primary_after = state("primary after cudaSetDevice(0)")

    ctx = torch.cuda.green_contexts.GreenContext.create(num_sms=40, device_id=0)
    ctx.set_context()
    green = state("green before")
    try:
        check("cudaSetDevice(green, same device)", RUNTIME.cudaSetDevice(0))
        after_first = state("green after cudaSetDevice(0)")
        # check("cudaSetDevice(green, repeated)", RUNTIME.cudaSetDevice(0))
        # after_second = state("after repeated cudaSetDevice(0)")
        print(
            "summary: "
            f"primary_unchanged={primary == primary_after} "
            f"green_differs_from_primary={green != primary} ",
            # f"same_device_selects_primary={after_first == primary == after_second}",
            flush=True,
        )
    finally:
        # Restore the Green context before its stack pop.
        check("cuCtxSetCurrent(green)", DRIVER.cuCtxSetCurrent(ctypes.c_void_p(green)))
        ctx.pop_context()
        state("after restore and pop")


if __name__ == "__main__":
    main()
