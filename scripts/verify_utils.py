import time
import torch
import numpy as np

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("[WARNING] psutil not installed.")
    print("Memory profiling disabled.")

def print_header(name):
    print("================================")
    print(f"VERIFY {name}")
    print("================================")

def print_footer(name, start_time, passed=True):
    elapsed = time.time() - start_time
    
    if PSUTIL_AVAILABLE:
        process = psutil.Process()
        cpu_mem = f"{process.memory_info().rss / 1024 / 1024:.2f} MB"
    else:
        cpu_mem = "N/A"
        
    gpu_mem = f"{torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB" if torch.cuda.is_available() else "0.00 MB"
    
    print("")
    print(f"Runtime       : {elapsed:.4f} sec")
    print(f"CPU Memory    : {cpu_mem}")
    if torch.cuda.is_available():
        print(f"GPU Memory    : {gpu_mem}")
        
    print("")
    if passed:
        print("PASS")
    else:
        print("FAIL")
        
    print(f"Report Saved: outputs/reports/{name}_REPORT.md")
    print("================================\n")

def assert_tensor_valid(tensor, name):
    assert not torch.isnan(tensor).any(), f"NaNs found in {name}"
    assert not torch.isinf(tensor).any(), f"Infs found in {name}"

def compute_entropy(prob_map):
    # prob_map: (B, 1, H, W)
    eps = 1e-8
    p = prob_map.view(prob_map.size(0), -1)
    entropy = -torch.sum(p * torch.log(p + eps), dim=1).mean()
    return entropy.item()

def print_feature_stats(name, tensor):
    print(f"{name}")
    print(f"  Shape : {list(tensor.shape)}")
    print(f"  Mean  : {tensor.mean().item():.4f}")
    print(f"  Std   : {tensor.std().item():.4f}")
    print(f"  Min   : {tensor.min().item():.4f}")
    print(f"  Max   : {tensor.max().item():.4f}")
    print("")
