import os
import shutil
import glob

def main():
    print("Preparing repository for GitHub submission...")
    
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # 1. Models directory
    models_dir = os.path.join(root_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    src_model = os.path.join(root_dir, 'outputs', 'checkpoints', 'best_model.pth')
    dst_model = os.path.join(models_dir, 'best_model.pth')
    
    if os.path.exists(src_model) and not os.path.exists(dst_model):
        shutil.copy2(src_model, dst_model)
        print(f"Copied best_model.pth to {models_dir}")
        
    # 2. Experimental scripts cleanup
    scripts_dir = os.path.join(root_dir, 'scripts')
    experimental_dir = os.path.join(scripts_dir, 'experimental')
    os.makedirs(experimental_dir, exist_ok=True)
    
    phase_scripts = glob.glob(os.path.join(scripts_dir, 'phase*.py'))
    ablation_scripts = glob.glob(os.path.join(scripts_dir, 'ablation*.py'))
    diagnose_scripts = glob.glob(os.path.join(scripts_dir, 'diagnose*.py'))
    
    experimental_scripts = phase_scripts + ablation_scripts + diagnose_scripts
    count = 0
    for script in experimental_scripts:
        basename = os.path.basename(script)
        if basename not in ['verify_submission.py', 'prepare_repo.py', 'train.py', 'generate_hackathon_dataset.py']:
            dst = os.path.join(experimental_dir, basename)
            if not os.path.exists(dst):
                shutil.move(script, dst)
                count += 1
                
    print(f"Moved {count} experimental scripts to scripts/experimental/")
    
    # 4. Generate requirements.txt
    import subprocess
    import sys
    req_path = os.path.join(root_dir, 'requirements.txt')
    try:
        with open(req_path, 'w') as f:
            subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=f, check=True)
        print(f"Generated requirements.txt at {req_path}")
    except Exception as e:
        print(f"Failed to generate requirements.txt: {e}")
        
    print("Repository preparation complete.")

if __name__ == "__main__":
    main()
