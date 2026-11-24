"""Map EGL devices to PCI buses and CUDA ordinals, to check which physical GPU MuJoCo renders on.

    .venv-libero/bin/python launch/egl_probe.py
"""
import ctypes
import os

egl = ctypes.CDLL("libEGL.so.1")
egl.eglGetProcAddress.restype = ctypes.c_void_p
egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]


def proc(name, restype, *argtypes):
    return ctypes.CFUNCTYPE(restype, *argtypes)(egl.eglGetProcAddress(name.encode()))


Dev = ctypes.c_void_p
query_devices = proc("eglQueryDevicesEXT", ctypes.c_uint, ctypes.c_int, ctypes.POINTER(Dev), ctypes.POINTER(ctypes.c_int))
query_string = proc("eglQueryDeviceStringEXT", ctypes.c_char_p, Dev, ctypes.c_int)
query_attrib = proc("eglQueryDeviceAttribEXT", ctypes.c_uint, Dev, ctypes.c_int, ctypes.POINTER(ctypes.c_ssize_t))
EGL_EXTENSIONS, EGL_DRM_DEVICE_FILE_EXT, EGL_CUDA_DEVICE_NV = 0x3055, 0x3233, 0x323A

devices, count = (Dev * 32)(), ctypes.c_int(0)
query_devices(32, devices, ctypes.byref(count))
print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} EGL devices={count.value}")
for i in range(count.value):
    drm = query_string(devices[i], EGL_DRM_DEVICE_FILE_EXT)
    pci = None
    if drm:
        card = os.path.basename(drm.decode())
        try:
            pci = os.path.basename(os.path.realpath(f"/sys/class/drm/{card}/device"))
        except OSError:
            pass
    cuda = ctypes.c_ssize_t(-1)
    ok = query_attrib(devices[i], EGL_CUDA_DEVICE_NV, ctypes.byref(cuda))
    ext = (query_string(devices[i], EGL_EXTENSIONS) or b"").decode()
    kind = "nvidia" if "EGL_NV_device_cuda" in ext else ("mesa" if "MESA" in ext else "other")
    print(f"  egl[{i}] {kind:6s} drm={drm.decode() if drm else None} pci={pci} cuda_ordinal={cuda.value if ok else 'n/a'}")
