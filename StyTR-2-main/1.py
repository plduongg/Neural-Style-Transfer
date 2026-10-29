import os
import sys
import subprocess
import torch

def run_cmd(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True, universal_newlines=True)
    except Exception as e:
        out = str(e)
    return out

print("Python executable:", sys.executable)
print("torch.__version__:", torch.__version__)
print("torch.version.cuda (built-with):", torch.version.cuda)
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.cuda.device_count():", torch.cuda.device_count())
try:
    if torch.cuda.is_available():
        print("current_device:", torch.cuda.current_device())
        print("get_device_name(0):", torch.cuda.get_device_name(0))
        print("memory_allocated (MB):", torch.cuda.memory_allocated(0)/1024**2)
        print("memory_reserved (MB):", torch.cuda.memory_reserved(0)/1024**2)
except Exception as e:
    print("cuda runtime error:", e)

print("torch.backends.cudnn.version():", torch.backends.cudnn.version())
try:
    from torch.utils.cpp_extension import CUDA_HOME
    print("CUDA_HOME (torch.utils.cpp_extension):", CUDA_HOME)
except Exception as e:
    print("CUDA_HOME check error:", e)

# Check nvcc (if installed)
print("\n--- nvcc --version ---")
print(run_cmd("nvcc --version"))

# On Windows, show relevant env vars
print("\n--- Environment variables ---")
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("PATH contains cuda:", any('CUDA' in p.upper() or 'NVIDIA' in p.upper() for p in os.environ.get("PATH","").split(os.pathsep)))

# nvidia-smi
print("\n--- nvidia-smi ---")
print(run_cmd("nvidia-smi -q -d COMPATIBILITY | findstr /R /C:\"Driver Version\" /C:\"CUDA Version\" || nvidia-smi"))