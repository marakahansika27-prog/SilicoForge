import time
import torch
import cv2
import numpy as np

# Phase 1
from src.preprocessing.ice import ImageConditioningEngine
from src.coarse_search.gspe import GlobalSearchProposalEngine
from src.feature_matching.gfee import GeometricFeatureExtractionEngine
from src.registration.srae import SpatialRegistrationAlignmentEngine
from src.localization.localization import ClassicalLocalization
# Phase 2
from src.ai_refinement.network import SNRN
from src.ai_refinement.config import SNRNConfig
# Phase 3
from src.integration.decision_fusion import DecisionFusionEngine

class HybridNavigationPipeline:
    def __init__(self, top_k: int = 1, nms_radius: int = None):
        # Initialize all required Phase 1, 2, and 3 engines
        self.ice = ImageConditioningEngine()
        
        # We sweep scale from 9.0 to 11.0, and rotation from -5 to 5.
        # This covers the hardest canonical dataset augmentations.
        scales = [9.0, 9.5, 10.0, 10.5, 11.0]
        rotations = [-5.0, -2.5, 0.0, 2.5, 5.0]
        self.gspe = GlobalSearchProposalEngine(
            top_k=top_k, 
            nms_radius=nms_radius,
            scale_hypotheses=scales,
            rotation_hypotheses=rotations
        )
        self.gfee = GeometricFeatureExtractionEngine()
        self.srae = SpatialRegistrationAlignmentEngine()
        self.loc = ClassicalLocalization()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.snrn = SNRN().to(self.device)
        
        # Load best model if exists
        import os
        checkpoint_path_models = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'models', 'best_model.pth')
        checkpoint_path_outputs = os.path.join("outputs", "checkpoints", "best_model.pth")
        
        checkpoint_path = checkpoint_path_models if os.path.exists(checkpoint_path_models) else checkpoint_path_outputs
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.snrn.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.snrn.load_state_dict(checkpoint)
            # FIX 1: Explicit success message for loading weights
            print(f"\n[INFO] Successfully loaded AI refinement weights from {checkpoint_path}")
        else:
            print("WARNING:\nRunning with randomly initialized SNRN.")
            
        self.snrn.eval()
        
        # FIX 6: Raise default confidence threshold and apply noise deadband
        self.fusion = DecisionFusionEngine(confidence_threshold=0.90, residual_deadband=0.10)
        
        self.state = {} # Holds diagnostics

    def run(self, ref_img, search_img):
        """
        Executes the entire hybrid pipeline end-to-end.
        """
        t0 = time.time()
        self.state = {
            'reference': ref_img,
            'search': search_img,
            'modules': {}
        }
        
        try:
            # 1. ICE
            cond = self.ice.run({'reference': ref_img, 'search': search_img})
            self.state['modules']['ICE'] = True
            
            # 2. GSPE
            gspe_res = self.gspe.run({'reference': cond['reference_cond'], 'search': cond['search_cond']})
            self.state['modules']['GSPE'] = True
            self.state['gspe_diagnostics'] = gspe_res
            
            if not gspe_res['boxes']:
                raise RuntimeError("GSPE found no candidates.")
                
            # USE GSPE TOP-1 FOR SNRN REFINEMENT
            # Retain top-k generation in GSPE for diagnostics, but only evaluate rank 1
            box = gspe_res['boxes'][0]
            x, y, w, h = box[0], box[1], box[2], box[3]
            top1_ncc = gspe_res['scores'][0] if 'scores' in gspe_res and len(gspe_res['scores']) > 0 else 0.0
            
            # Use integer indices for cropping
            cand_crop = cond['search_cond'][int(y):int(y+h), int(x):int(x+w)]
            center_x = float(x) + float(w) / 2.0
            center_y = float(y) + float(h) / 2.0
            
            classical_coord = np.array([center_x, center_y], dtype=np.float32)
            
            print(f"\n--- GSPE TOP-1 SELECTED ---")
            print(f"GSPE Coordinate : ({classical_coord[0]:.2f}, {classical_coord[1]:.2f})")
            print(f"GSPE Score      : {top1_ncc:.4f}")
            
            def center_crop(img, size):
                h_img, w_img = img.shape
                ch, cw = size, size
                y_start = max(0, h_img//2 - ch//2)
                x_start = max(0, w_img//2 - cw//2)
                crop = img[y_start:y_start+ch, x_start:x_start+cw]
                if crop.shape[0] < ch or crop.shape[1] < cw:
                    crop = cv2.copyMakeBorder(crop, 0, ch-crop.shape[0], 0, cw-crop.shape[1], cv2.BORDER_CONSTANT, value=0)
                return crop
                
            ref_full = cond['reference_cond']
            scale_ratio = 10.0 # Match GSPE bounding box scale
            w_ref, h_ref = ref_full.shape[1], ref_full.shape[0]
            scaled_w = int(round(w_ref / scale_ratio))
            scaled_h = int(round(h_ref / scale_ratio))
            ref_scaled = cv2.resize(ref_full, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
            
            ref_patch_np = center_crop(ref_scaled, SNRNConfig.PATCH_SIZE).astype(np.float32) / 255.0
            ref_tensor = torch.from_numpy(ref_patch_np).unsqueeze(0).unsqueeze(0).to(self.device)
            
            local_class_x = classical_coord[0] - x
            local_class_y = classical_coord[1] - y
            
            M_extract = np.array([
                [1.0, 0.0, (SNRNConfig.PATCH_SIZE / 2.0) - local_class_x],
                [0.0, 1.0, (SNRNConfig.PATCH_SIZE / 2.0) - local_class_y]
            ], dtype=np.float32)
            
            cand_patch_raw = cv2.warpAffine(
                cand_crop, M_extract, 
                (SNRNConfig.PATCH_SIZE, SNRNConfig.PATCH_SIZE), 
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
            )
            cand_patch_np = cand_patch_raw.astype(np.float32) / 255.0
            cand_tensor = torch.from_numpy(cand_patch_np).unsqueeze(0).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                ai_preds = self.snrn(ref_tensor, cand_tensor)
                
            # Record state
            self.state['modules']['GFEE'] = False
            self.state['modules']['SRAE'] = False
            self.state['modules']['Localization'] = False
            self.state['coarse_box'] = box
            self.state['reference_patch'] = ref_patch_np
            self.state['rectified_patch'] = cand_patch_np
            self.state['modules']['AI Refinement'] = True
            
            res_dx = float(ai_preds['residual'][0, 0].detach().cpu().item())
            res_dy = float(ai_preds['residual'][0, 1].detach().cpu().item())
            confidence = float(ai_preds['confidence'][0].detach().cpu().item())
            res_mag = float(np.sqrt(res_dx**2 + res_dy**2))
            
            if np.isnan(res_dx) or np.isnan(res_dy) or np.isinf(res_dx) or np.isinf(res_dy):
                raise ValueError(f"AI predicted invalid residual coordinates: ({res_dx}, {res_dy})")
            if np.isnan(confidence) or np.isinf(confidence):
                raise ValueError(f"AI predicted invalid confidence: {confidence}")
            
            print(f"\n--- AI PREDICTION ---")
            print(f"Residual dx        : {res_dx:.4f} px")
            print(f"Residual dy        : {res_dy:.4f} px")
            print(f"Residual Magnitude : {res_mag:.4f} px")
            print(f"Confidence Score   : {confidence:.4f}")
            print(f"---------------------\n")
            
            # SAFETY GATE FUSION
            # Apply a strict residual safety gate based on residual magnitude
            max_allowed_res = getattr(SNRNConfig, 'MAX_CLASSICAL_ERROR', 5.0)
            conf_threshold = 0.90
            
            ai_coord = np.array([classical_coord[0] + res_dx, classical_coord[1] + res_dy], dtype=np.float32)
            
            print(f"\n--- DECISION FUSION ---")
            print(f"Classical : [{classical_coord[0]:.4f}, {classical_coord[1]:.4f}]")
            print(f"AI        : [{ai_coord[0]:.4f}, {ai_coord[1]:.4f}]")
            print(f"Residual  : [{res_dx:.4f}, {res_dy:.4f}]")
            print(f"Magnitude : {res_mag:.4f} px")
            print(f"Confidence: {confidence:.4f}")
            print(f"Max Allow : {max_allowed_res:.4f} px")
            
            if res_mag <= max_allowed_res and confidence >= conf_threshold:
                final_coord = ai_coord
                decision = "AI_REFINED"
                self.state['modules']['Fusion'] = True
            else:
                final_coord = classical_coord
                decision = "CLASSICAL_GSPE_FALLBACK"
                self.state['modules']['Fusion'] = False
                
            print(f"Decision  : {decision}")
            
            # Distance from GSPE coord
            dist_from_gspe = float(np.linalg.norm(final_coord - classical_coord))
            
            self.state['gspe_selected_rank'] = 1
            self.state['gspe_selected_score'] = top1_ncc
            self.state['classical_coord'] = classical_coord
            self.state['ai_coord'] = ai_coord
            self.state['final_coord'] = final_coord
            self.state['confidence'] = confidence
            self.state['decision'] = decision
            self.state['dist_from_gspe'] = dist_from_gspe
            self.state['ai_residual'] = np.array([res_dx, res_dy], dtype=np.float32)
            self.state['ai_residual_mag'] = res_mag
            self.state['runtime'] = time.time() - t0
            
            return self.state
            
        except Exception as e:
            self.state['error'] = str(e)
            self.state['runtime'] = time.time() - t0
            return self.state
