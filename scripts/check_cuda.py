import torch
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
    print("Device Count:", torch.cuda.device_count())
    print("VRAM (GB):", torch.cuda.get_device_properties(0).total_memory / 1024**3)
