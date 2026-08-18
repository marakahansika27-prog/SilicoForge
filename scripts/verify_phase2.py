import os
import sys
import subprocess
import time

def main():
    print("\nStarting Phase 2 Master Verification Suite...\n")
    
    scripts = [
        ("Dataset", "verify_dataset.py"),
        ("Network", "verify_network.py"),
        ("Forward", "verify_forward.py"),
        ("Loss", "verify_loss.py"),
        ("Gradients", "verify_gradient_flow.py"),
        ("Training", "verify_training.py"),
        ("Inference", "verify_inference.py")
    ]
    
    results = {}
    overall_pass = True
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    os.makedirs(os.path.join(script_dir, "..", "outputs", "reports"), exist_ok=True)
    with open(os.path.join(script_dir, "..", "outputs", "reports", "PHASE2_REPORT.md"), "w") as report_file:
        report_file.write("# Phase 2 Master Verification Report\n\n")
        
        for name, script_file in scripts:
            script_path = os.path.join(script_dir, script_file)
            print(f"Executing {script_file}...")
            
            start_time = time.time()
            
            # Using sys.executable to ensure we use the current Python environment
            result = subprocess.run([sys.executable, script_path], 
                                    capture_output=True, text=True)
            
            elapsed = time.time() - start_time
            
            # Print the console output of the sub-script directly
            print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            
            passed = (result.returncode == 0)
            results[name] = passed
            if not passed:
                overall_pass = False
                
            report_file.write(f"## {name} Verification\n")
            report_file.write(f"- Status: **{'PASS' if passed else 'FAIL'}**\n")
            report_file.write(f"- Runtime: {elapsed:.2f}s\n\n")
            
        report_file.write(f"## OVERALL STATUS: **{'PASS' if overall_pass else 'FAIL'}**\n")
            
    print("\n==============================")
    print("PHASE 2 VERIFICATION")
    print("==============================")
    for name, script_file in scripts:
        status = "PASS" if results[name] else "FAIL"
        print(f"{name:<15} ........ {status}")
    print("==============================")
    print("OVERALL STATUS")
    print("==============================")
    print("PASS" if overall_pass else "FAIL")
    print("==============================")
    print("Generated PHASE2_REPORT.md")
    
if __name__ == "__main__":
    main()
