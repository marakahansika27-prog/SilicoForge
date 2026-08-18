import os
import sys
import subprocess
import json
import hashlib

def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    def check_file(path):
        return os.path.exists(os.path.join(root, path))
        
    print("========================================")
    print("DRIFT-SENSE V2 GITHUB SUBMISSION AUDIT")
    print("========================================\n")

    readme = check_file("README.md")
    print(f"README:\n{'PASS' if readme else 'FAIL'}\n")
    
    gen = check_file("generate_dataset.py")
    print(f"Dataset generator:\n{'PASS' if gen else 'FAIL'}\n")
    
    inf = check_file("inference.py")
    print(f"Inference script:\n{'PASS' if inf else 'FAIL'}\n")
    
    model = check_file("models/best_model.pth")
    print(f"Model artifact:\n{'PASS' if model else 'FAIL'}\n")
    
    train = check_file("scripts/train.py")
    print(f"Training script:\n{'PASS' if train else 'FAIL'}\n")
    
    req = check_file("requirements.txt")
    print(f"requirements.txt:\n{'PASS' if req else 'FAIL'}\n")
    
    ref = check_file("docs/references.md")
    print(f"References:\n{'PASS' if ref else 'FAIL'}\n")
    
    pipe = check_file("docs/pipeline.md")
    print(f"Pipeline documentation:\n{'PASS' if pipe else 'FAIL'}\n")
    
    # We simulate the fresh machine test programmatically
    print("Running Fresh-Machine Test...")
    test_dir = os.path.join(root, "fresh_test")
    
    # Generate DRAM
    cmd_dram = [sys.executable, os.path.join(root, "generate_dataset.py"), "--architecture", "DRAM", "--num-pairs", "1", "--output-dir", test_dir]
    res_dram = subprocess.run(cmd_dram, capture_output=True)
    dram_gen_pass = res_dram.returncode == 0
    print(f"DRAM generation:\n{'PASS' if dram_gen_pass else 'FAIL'}\n")
    
    # Generate FinFET
    cmd_finfet = [sys.executable, os.path.join(root, "generate_dataset.py"), "--architecture", "FinFET", "--num-pairs", "1", "--output-dir", test_dir]
    res_finfet = subprocess.run(cmd_finfet, capture_output=True)
    finfet_gen_pass = res_finfet.returncode == 0
    print(f"FinFET generation:\n{'PASS' if finfet_gen_pass else 'FAIL'}\n")
    
    # GT recording
    dram_gt = os.path.exists(os.path.join(test_dir, "dram", "pair_0001", "ground_truth.json"))
    finfet_gt = os.path.exists(os.path.join(test_dir, "finfet", "pair_0001", "ground_truth.json"))
    gt_record_pass = dram_gt and finfet_gt
    print(f"Ground truth recording:\n{'PASS' if gt_record_pass else 'FAIL'}\n")
    
    # Inference DRAM
    dram_inf_pass = False
    if dram_gt:
        cmd_inf = [sys.executable, os.path.join(root, "inference.py"), os.path.join(test_dir, "dram", "pair_0001", "reference.png"), os.path.join(test_dir, "dram", "pair_0001", "search.png")]
        res_inf = subprocess.run(cmd_inf, capture_output=True, text=True)
        if res_inf.returncode == 0 and "(" in res_inf.stdout and ")" in res_inf.stdout:
            dram_inf_pass = True
    print(f"Inference on DRAM:\n{'PASS' if dram_inf_pass else 'FAIL'}\n")
    
    # Inference FinFET
    finfet_inf_pass = False
    if finfet_gt:
        cmd_inf = [sys.executable, os.path.join(root, "inference.py"), os.path.join(test_dir, "finfet", "pair_0001", "reference.png"), os.path.join(test_dir, "finfet", "pair_0001", "search.png")]
        res_inf = subprocess.run(cmd_inf, capture_output=True, text=True)
        if res_inf.returncode == 0 and "(" in res_inf.stdout and ")" in res_inf.stdout:
            finfet_inf_pass = True
    print(f"Inference on FinFET:\n{'PASS' if finfet_inf_pass else 'FAIL'}\n")
    
    # Check GT leakage
    gt_leakage = False
    with open(os.path.join(root, "inference.py"), 'r') as f:
        content = f.read()
        if "ground_truth.json" in content or "metadata.json" in content:
            gt_leakage = True
    
    with open(os.path.join(root, "src", "integration", "pipeline_backup_v2_ai.py"), 'r') as f:
        content = f.read()
        if "ground_truth.json" in content or "metadata.json" in content:
            gt_leakage = True
            
    print(f"No GT leakage:\n{'PASS' if not gt_leakage else 'FAIL'}\n")
    
    visual_pass = os.path.exists(os.path.join(root, "outputs", "hackathon_v2", "phase104_visuals"))
    print(f"Visual outputs:\n{'PASS' if visual_pass else 'FAIL'}\n")
    
    github_pass = check_file(".gitignore") and os.path.exists(os.path.join(root, "scripts", "experimental"))
    print(f"GitHub structure:\n{'PASS' if github_pass else 'FAIL'}\n")
    
    fresh_machine = dram_gen_pass and finfet_gen_pass and gt_record_pass and dram_inf_pass and finfet_inf_pass
    print(f"Fresh-machine test:\n{'PASS' if fresh_machine else 'FAIL'}\n")
    
    all_pass = (readme and gen and inf and model and train and req and ref and pipe and fresh_machine and not gt_leakage and github_pass)
    
    print("Overall submission readiness:")
    print("READY" if all_pass else "NOT READY")
    print("========================================")

if __name__ == "__main__":
    main()
