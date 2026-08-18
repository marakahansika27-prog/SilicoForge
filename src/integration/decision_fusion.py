import numpy as np

class DecisionFusionEngine:
    """
    Implements confidence-based and deadband-gated decision fusion for navigation coordinates.
    If the AI Refinement Network is confident AND its predicted residual is larger
    than the intrinsic noise floor (deadband), its sub-pixel correction is used.
    Otherwise, the pipeline falls back to the classical coordinate.
    """
    def __init__(self, confidence_threshold=0.90, residual_deadband=0.10):
        self.confidence_threshold = confidence_threshold
        self.residual_deadband = residual_deadband
        
    def run(self, inputs):
        ai_coord = np.array(inputs['ai_coord'], dtype=np.float32)
        classical_coord = np.array(inputs['classical_coord'], dtype=np.float32)
        confidence = float(inputs.get('confidence', 0.0))
        
        # 1. Safety Checks
        safe = True
        residual_mag = 0.0
        
        if np.isnan(confidence) or np.isinf(confidence):
            safe = False
            
        if safe and (np.isnan(ai_coord).any() or np.isinf(ai_coord).any() or 
                     np.isnan(classical_coord).any() or np.isinf(classical_coord).any()):
            safe = False
            
        if safe and ai_coord.shape != classical_coord.shape:
            safe = False
            
        if safe:
            residual_vector = ai_coord - classical_coord
            residual_mag = np.linalg.norm(residual_vector)
            if np.isnan(residual_mag) or np.isinf(residual_mag):
                safe = False
                
        # 2. Logic
        if safe and confidence >= self.confidence_threshold and residual_mag > self.residual_deadband:
            final_coord = ai_coord
            decision = "AI_REFINED"
        else:
            final_coord = classical_coord
            decision = "CLASSICAL_FALLBACK"
            
        # Logging
        if safe:
            res_dx, res_dy = residual_vector
        else:
            res_dx, res_dy = 0.0, 0.0
            
        print("\n--- DECISION FUSION ---")
        print(f"Classical : [{classical_coord[0]:.4f}, {classical_coord[1]:.4f}]")
        print(f"AI        : [{ai_coord[0]:.4f}, {ai_coord[1]:.4f}]")
        print(f"Residual  : [{res_dx:.4f}, {res_dy:.4f}]")
        print(f"Magnitude : {residual_mag:.4f} px")
        print(f"Confidence: {confidence:.4f}")
        print(f"Deadband  : {self.residual_deadband:.4f} px")
        print(f"Decision  : {decision}")
            
        return {
            'final_coordinate': final_coord,
            'decision': decision,
            'confidence': confidence
        }
