import os

def find_pth_files(root_dir):
    pth_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.pth') or filename.endswith('.pt'):
                full_path = os.path.join(dirpath, filename)
                stat = os.stat(full_path)
                pth_files.append({
                    'path': full_path,
                    'size': stat.st_size,
                    'mtime': stat.st_mtime
                })
    return pth_files

if __name__ == "__main__":
    out_dir = "C:/d-s/Drift-Sense-V2/outputs"
    files = find_pth_files(out_dir)
    print("Found .pth files:")
    for f in files:
        print(f"Path: {f['path']} | Size: {f['size']} | Mtime: {f['mtime']}")
