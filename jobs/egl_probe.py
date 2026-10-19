"""List EGL devices visible inside this SLURM allocation and whether each initialises (robosuite renders via EGL)."""
import ctypes, os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
from OpenGL import EGL
from OpenGL.EGL.EXT.device_query import eglQueryDeviceStringEXT
from OpenGL.EGL.EXT.platform_device import EGL_PLATFORM_DEVICE_EXT
from OpenGL.EGL.EXT.device_enumeration import eglQueryDevicesEXT
num = EGL.EGLint()
eglQueryDevicesEXT(0, None, ctypes.pointer(num))
devs = (EGL.EGLDeviceEXT * num.value)()
eglQueryDevicesEXT(num.value, devs, ctypes.pointer(num))
print(f"EGL devices: {num.value}  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}  "
      f"SLURM_JOB_GPUS={os.environ.get('SLURM_JOB_GPUS')}  MUJOCO_EGL_DEVICE_ID={os.environ.get('MUJOCO_EGL_DEVICE_ID')}")
for i in range(num.value):
    try:
        disp = EGL.eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, devs[i], None)
        ok = bool(EGL.eglInitialize(disp, None, None))
        if ok: EGL.eglTerminate(disp)
    except Exception as e:  # noqa: BLE001
        ok = f"error: {e}"
    print(f"  egl[{i}] init={ok}")
