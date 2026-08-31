import os
import json
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

class Phase2EvaluationDataset(Dataset):
    """
    Dedicated evaluation dataset for Phase 2 real generated cases.
    Scans a directory for case subdirectories containing:
      - reference.png
      - search.png
      - metadata.json
      
    Returns preprocessed tensors matching Phase 1 inference pipeline conventions.
    """
    def __init__(self, dataset_dir):
        # Explicitly resolve absolute path
        abs_dataset_dir = os.path.abspath(dataset_dir)
        print("DATASET ROOT")
        print("------------")
        print(f"Requested : {dataset_dir}")
        print(f"Resolved  : {abs_dataset_dir}")
        print(f"Exists    : {os.path.exists(abs_dataset_dir)}\n")
        
        self.dataset_dir = abs_dataset_dir
        self.cases = []
        
        # Discover cases
        if not os.path.exists(abs_dataset_dir):
            raise ValueError(f"Dataset directory not found: {abs_dataset_dir}")
            
        for arch in ['dram', 'finfet']:
            arch_dir = os.path.join(abs_dataset_dir, arch)
            if not os.path.isdir(arch_dir):
                continue
                
            for case_name in sorted(os.listdir(arch_dir)):
                case_dir = os.path.join(arch_dir, case_name)
                if not os.path.isdir(case_dir):
                    continue
                    
                meta_path = os.path.join(case_dir, "metadata.json")
                ref_path = os.path.join(case_dir, "reference.png")
                search_path = os.path.join(case_dir, "search.png")
                
                if os.path.exists(meta_path) and os.path.exists(ref_path) and os.path.exists(search_path):
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    
                    # Contract mapping
                    gt_x = meta.get('gt_x', -1.0)
                    gt_y = meta.get('gt_y', -1.0)
                    target_present = meta.get('target_present', (gt_x >= 0 and gt_y >= 0))
                    
                    self.cases.append({
                        'case_dir': case_dir,
                        'case_id': meta.get('case_id', case_name),
                        'architecture': meta.get('architecture', arch.upper()),
                        'target_present': target_present,
                        'gt_x': float(gt_x) if target_present else -1.0,
                        'gt_y': float(gt_y) if target_present else -1.0,
                        'gt_bbox': meta.get('gt_bbox', {}),
                        'phase2_scale': float(meta.get('effective_scale', meta.get('nominal_scale', 10.0))),
                        'rotation_degrees': float(meta.get('rotation_degrees', 0.0))
                    })
                    
        self.positive_count = sum(1 for c in self.cases if c['target_present'])
        self.negative_count = len(self.cases) - self.positive_count

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case_info = self.cases[idx]
        
        # Load images
        ref_path = os.path.join(case_info['case_dir'], "reference.png")
        search_path = os.path.join(case_info['case_dir'], "search.png")
        
        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
        
        # We must return the raw loaded images and metadata so the verifier can pass them through ICE -> GSPE -> SNRN
        
        return {
            'reference_img': ref_img,
            'search_img': search_img,
            'case_info': case_info
        }
