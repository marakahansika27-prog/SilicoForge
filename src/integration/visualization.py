import os
import cv2
import numpy as np
import torch

class PipelineVisualizer:
    """
    Generates and saves the complete set of visual diagnostics for Phase 3 integration.
    """
    def __init__(self, output_dir="outputs/pipeline"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _save(self, img, filename):
        if isinstance(img, torch.Tensor):
            img = img.squeeze().detach().cpu().numpy()
            if img.max() <= 1.01:
                img = (img * 255).clip(0, 255)
        path = os.path.join(self.output_dir, filename)
        cv2.imwrite(path, img.astype(np.uint8))
        
    def run(self, pipeline_data):
        """
        Saves all requested diagnostic images based on pipeline dictionary state.
        """
        if 'reference' in pipeline_data:
            self._save(pipeline_data['reference'], "reference.png")
            
        if 'search' in pipeline_data:
            search = pipeline_data['search'].copy()
            # Draw Coarse Match box if available
            if 'coarse_box' in pipeline_data:
                x, y, w, h = pipeline_data['coarse_box']
                # Convert grayscale to BGR for drawing colored box
                if len(search.shape) == 2:
                    search = cv2.cvtColor((search*255 if search.max()<=1.01 else search).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                cv2.rectangle(search, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 255), 2)
                cv2.imwrite(os.path.join(self.output_dir, "coarse_match.png"), search)
            else:
                self._save(search, "candidate.png") # original search
                
        # Draw Feature Matches
        if 'match_img' in pipeline_data:
            self._save(pipeline_data['match_img'], "feature_matches.png")
            
        if 'rectified_patch' in pipeline_data:
            self._save(pipeline_data['rectified_patch'], "registered.png")
            
        # Draw Difference Map
        if 'reference_patch' in pipeline_data and 'rectified_patch' in pipeline_data:
            ref = pipeline_data['reference_patch']
            cand = pipeline_data['rectified_patch']
            if isinstance(ref, np.ndarray) and isinstance(cand, np.ndarray):
                # Ensure same size
                rh, rw = ref.shape[:2]
                ch, cw = cand.shape[:2]
                if rh == ch and rw == cw:
                    diff = np.abs(ref - cand)
                    self._save(diff, "difference_map.png")
            
        # Save AI outputs
        if 'ai_heatmap' in pipeline_data:
            hm = pipeline_data['ai_heatmap']
            if isinstance(hm, torch.Tensor):
                hm = hm.squeeze().detach().cpu().numpy()
            hm_norm = (hm / (hm.max() + 1e-8)) * 255
            heatmap_color = cv2.applyColorMap(hm_norm.astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(self.output_dir, "heatmap.png"), heatmap_color)
            
        if 'ai_attention' in pipeline_data:
            attn = pipeline_data['ai_attention']
            if isinstance(attn, torch.Tensor):
                attn = attn.mean(dim=0).view(8, 8).detach().cpu().numpy()
            attn_resized = cv2.resize(attn, (128, 128))
            attn_norm = (attn_resized / (attn_resized.max() + 1e-8)) * 255
            attn_color = cv2.applyColorMap(attn_norm.astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(self.output_dir, "attention_map.png"), attn_color)
            
        # Final Prediction vs GT visualization
        if 'final_coord' in pipeline_data and 'ground_truth' in pipeline_data and 'search' in pipeline_data:
            s_img = pipeline_data['search'].copy()
            if len(s_img.shape) == 2:
                s_img = cv2.cvtColor((s_img*255 if s_img.max()<=1.01 else s_img).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                
            fx, fy = pipeline_data['final_coord']
            gx, gy = pipeline_data['ground_truth']
            
            # Draw GT (Green) and Final (Red)
            cv2.circle(s_img, (int(gx), int(gy)), 5, (0, 255, 0), -1)
            cv2.circle(s_img, (int(fx), int(fy)), 5, (0, 0, 255), -1)
            cv2.line(s_img, (int(gx), int(gy)), (int(fx), int(fy)), (255, 0, 0), 2)
            cv2.imwrite(os.path.join(self.output_dir, "final_prediction.png"), s_img)
