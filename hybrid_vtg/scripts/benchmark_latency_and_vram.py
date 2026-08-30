#!/usr/bin/env python3
import time
import torch

def profile_hardware():
    if not torch.cuda.is_available():
        print("CUDA unavailable, skipping hardware profiling.")
        return

    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    print(f"\n=== HARDWARE PROFILING REPORT ({device_name}) ===")
    print(f"Total VRAM: {total_vram:.2f} GB")
    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    reserved = torch.cuda.memory_reserved(0) / (1024**3)
    print(f"Current Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

if __name__ == "__main__":
    profile_hardware()
