"""Quick VRAM / memory check for benchmarking."""
import torch
import psutil

print(f"PyTorch: {torch.__version__}")
print(f"ROCm available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    total = torch.cuda.get_device_properties(0).total_memory
    print(f"Total VRAM: {total / 1e9:.2f} GB ({total / (1024**3):.2f} GiB)")
    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    print(f"Allocated: {allocated / 1e9:.2f} GB")
    print(f"Reserved:  {reserved / 1e9:.2f} GB")
    print(f"Free VRAM: {(total - reserved) / 1e9:.2f} GB")

print()
mem = psutil.virtual_memory()
print(f"System RAM total:     {mem.total / 1e9:.2f} GB")
print(f"System RAM available: {mem.available / 1e9:.2f} GB")
print(f"System RAM used:      {mem.used / 1e9:.2f} GB")
