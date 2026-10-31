import ctypes
import os
import time

DLL = r"C:\Program Files (x86)\VB\Voicemeeter\VoicemeeterRemote64.dll"


def ensure_banana_routing():
    if not os.path.exists(DLL):
        return False, "VoiceMeeter Remote API not found"

    vm = ctypes.WinDLL(DLL)
    vm.VBVMR_Login.restype = ctypes.c_long
    vm.VBVMR_Logout.restype = ctypes.c_long
    vm.VBVMR_RunVoicemeeter.argtypes = [ctypes.c_long]
    vm.VBVMR_RunVoicemeeter.restype = ctypes.c_long
    vm.VBVMR_IsParametersDirty.restype = ctypes.c_long
    vm.VBVMR_SetParameterFloat.argtypes = [ctypes.c_char_p, ctypes.c_float]
    vm.VBVMR_SetParameterFloat.restype = ctypes.c_long

    login = vm.VBVMR_Login()
    if login < 0:
        return False, f"VoiceMeeter login error {login}"
    if login == 1:
        run = vm.VBVMR_RunVoicemeeter(2)
        if run < 0:
            return False, f"VoiceMeeter start error {run}"
        time.sleep(2.0)

    for _ in range(3):
        vm.VBVMR_IsParametersDirty()
        time.sleep(0.05)

    settings = {
        "Strip[3].A1": 0, "Strip[3].A2": 0, "Strip[3].A3": 0,
        "Strip[3].B1": 1, "Strip[3].B2": 0,
        "Strip[4].A1": 0, "Strip[4].A2": 0, "Strip[4].A3": 0,
        "Strip[4].B1": 0, "Strip[4].B2": 1,
    }
    for name, value in settings.items():
        result = vm.VBVMR_SetParameterFloat(name.encode("ascii"), ctypes.c_float(value))
        if result < 0:
            vm.VBVMR_Logout()
            return False, f"Routing error {name}: {result}"

    time.sleep(0.5)
    for _ in range(4):
        vm.VBVMR_IsParametersDirty()
        time.sleep(0.05)
    vm.VBVMR_Logout()
    return True, "VoiceMeeter routing ready"
