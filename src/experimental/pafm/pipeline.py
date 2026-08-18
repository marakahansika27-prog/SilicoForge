import time
import numpy as np
import io
import contextlib
from src.preprocessing.ice import ImageConditioningEngine
from src.coarse_search.gspe import GlobalSearchProposalEngine
from src.experimental.pafm.template_family import generate_family_variants
from src.experimental.pafm.resolution_match import match_resolution
from src.experimental.pafm.peak_extraction import extract_diagnostic_peaks

class PAFMExperimentalPipeline:
    """
    PAFM Phase 1 Pipeline:
    REFERENCE -> TEMPLATE FAMILY -> RESOLUTION MATCHING -> EXISTING GSPE -> CORRELATION SURFACES -> DIAGNOSTIC PEAKS
    """
    def __init__(self, top_k=20, sigma=1.0, diagnostic_radius=50):
        self.top_k = top_k
        self.sigma = sigma
        self.diagnostic_radius = diagnostic_radius
        self.ice = ImageConditioningEngine()
        # Ensure scale=10.0 and rotation=0.0 are strictly preserved per the baseline
        self.gspe = GlobalSearchProposalEngine(top_k=1, scale_hypotheses=[10.0], rotation_hypotheses=[0.0])
        self.scale = 10.0
        
    def run(self, ref_img, search_img, ref_macro):
        t0 = time.time()
        
        # 1. Base ICE conditioning on macro template and search
        # Note: we use ref_macro instead of ref_img for family generation, 
        # as it is the template used in the baseline coarse search.
        cond = self.ice.run({"reference": ref_macro, "search": search_img})
        search_cond = cond["search_cond"]
        ref_cond = cond["reference_cond"] # This is the ICE-conditioned ref_macro
        
        # 2. Resolution Matching (Search Image) - Cache once
        search_matched = match_resolution(search_cond, sigma=self.sigma)
        
        # 3. Template Family Generation
        # Generating 25 template variants by shifting the reference in its own coordinate space.
        variants = generate_family_variants(ref_cond, 
                                            offsets_x=[-2, -1, 0, 1, 2], 
                                            offsets_y=[-2, -1, 0, 1, 2])
        
        family_results = []
        all_peaks = []
        
        for variant in variants:
            # 4. Resolution Matching (Template Variant)
            template_matched = match_resolution(variant["template"], sigma=self.sigma)
            
            # 5. Existing GSPE Correlation
            with contextlib.redirect_stdout(io.StringIO()):
                gspe_res = self.gspe.run({
                    "reference": template_matched,
                    "search": search_matched
                })
                
            res_hybrid = gspe_res["res_hybrid"]
            
            # 6. Diagnostic Peak Extraction
            peaks = extract_diagnostic_peaks(res_hybrid, num_peaks=self.top_k, radius=self.diagnostic_radius)
            
            # 7. Canonical Coordinate Correction
            # Transform raw peaks to canonical coordinates by compensating for the template offset.
            # If the template was shifted by (dx, dy) in reference space (moving the content RIGHT/DOWN),
            # the corresponding peak in the search space shifts LEFT/UP by (-dx/scale, -dy/scale).
            # To recover canonical baseline anchor coordinates, we ADD (dx/scale, dy/scale) back.
            dx = variant["metadata"]["offset_x"]
            dy = variant["metadata"]["offset_y"]
            
            canonical_peaks = []
            for cx, cy, score in peaks:
                canonical_cx = cx + (dx / self.scale)
                canonical_cy = cy + (dy / self.scale)
                canonical_peaks.append((canonical_cx, canonical_cy, score))
                
                all_peaks.append({
                    "canonical_x": canonical_cx,
                    "canonical_y": canonical_cy,
                    "raw_x": cx,
                    "raw_y": cy,
                    "score": score,
                    "family_id": variant["metadata"]["family_id"],
                    "offset_x": dx,
                    "offset_y": dy
                })
                
            # Store family-level results
            variant["metadata"]["best_canonical_x"] = canonical_peaks[0][0] if canonical_peaks else -1.0
            variant["metadata"]["best_canonical_y"] = canonical_peaks[0][1] if canonical_peaks else -1.0
            variant["metadata"]["best_score"] = canonical_peaks[0][2] if canonical_peaks else -999.0
            
            family_results.append({
                "metadata": variant["metadata"],
                "canonical_peaks": canonical_peaks,
                "res_hybrid": res_hybrid
            })
            
        # 8. Global Candidate Assembly
        # Sort all peaks across families by score to find the global Top-1
        all_peaks.sort(key=lambda p: p["score"], reverse=True)
        
        # Spatial NMS across the global pool to get spatially distinct Top-K diagnostic candidates
        top_candidates = []
        for p in all_peaks:
            is_distinct = True
            for tc in top_candidates:
                dist = np.sqrt((p["canonical_x"] - tc["canonical_x"])**2 + (p["canonical_y"] - tc["canonical_y"])**2)
                if dist < self.diagnostic_radius:
                    is_distinct = False
                    break
            if is_distinct:
                top_candidates.append(p)
            if len(top_candidates) >= self.top_k:
                break
                
        best_family = max(family_results, key=lambda f: f["metadata"]["best_score"])
        
        return {
            "family_results": family_results,
            "best_family_id": best_family["metadata"]["family_id"],
            "best_score": best_family["metadata"]["best_score"],
            "best_location": (best_family["metadata"]["best_canonical_x"], best_family["metadata"]["best_canonical_y"]),
            "candidate_peaks": top_candidates, # Canonically corrected, spatially distinct top diagnostic candidates
            "all_family_peaks": all_peaks,
            "runtime": time.time() - t0,
            "search_matched": search_matched # Preserved for visualizations
        }

from src.experimental.pafm.candidate_aggregation import cluster_candidates
from src.experimental.pafm.family_voting import calculate_family_support
from src.experimental.pafm.lattice_analysis import lattice_aware_nms
from src.experimental.pafm.ambiguity import classify_ambiguity
from src.experimental.pafm.structural_selection import select_candidate

class PAFMPhase2Pipeline:
    def __init__(self, top_k=20, sigma=1.0, diagnostic_radius=50, support_radius=5.0):
        self.top_k = top_k
        self.sigma = sigma
        self.diagnostic_radius = diagnostic_radius
        self.support_radius = support_radius
        self.ice = ImageConditioningEngine()
        self.gspe = GlobalSearchProposalEngine(top_k=1, scale_hypotheses=[10.0], rotation_hypotheses=[0.0])
        self.scale = 10.0
        
    def run(self, ref_img, search_img, ref_macro, ref_100):
        t0 = time.time()
        
        # 1. Base ICE conditioning
        cond = self.ice.run({"reference": ref_macro, "search": search_img})
        search_cond = cond["search_cond"]
        ref_cond = cond["reference_cond"]
        
        # 2. Resolution Matching
        search_matched = match_resolution(search_cond, sigma=self.sigma)
        
        # 3. Template Family Generation
        variants = generate_family_variants(ref_cond, 
                                            offsets_x=[-2, -1, 0, 1, 2], 
                                            offsets_y=[-2, -1, 0, 1, 2])
        
        all_peaks = []
        family_results = []
        
        for variant in variants:
            template_matched = match_resolution(variant["template"], sigma=self.sigma)
            
            with contextlib.redirect_stdout(io.StringIO()):
                gspe_res = self.gspe.run({
                    "reference": template_matched,
                    "search": search_matched
                })
                
            res_hybrid = gspe_res["res_hybrid"]
            peaks = extract_diagnostic_peaks(res_hybrid, num_peaks=self.top_k, radius=self.diagnostic_radius)
            
            dx = variant["metadata"]["offset_x"]
            dy = variant["metadata"]["offset_y"]
            
            canonical_peaks = []
            for cx, cy, score in peaks:
                canonical_cx = cx + (dx / self.scale)
                canonical_cy = cy + (dy / self.scale)
                canonical_peaks.append((canonical_cx, canonical_cy, score))
                all_peaks.append({
                    "canonical_x": canonical_cx,
                    "canonical_y": canonical_cy,
                    "raw_x": cx,
                    "raw_y": cy,
                    "score": score,
                    "family_id": variant["metadata"]["family_id"],
                    "offset_x": dx,
                    "offset_y": dy
                })
            variant["metadata"]["best_canonical_x"] = canonical_peaks[0][0] if canonical_peaks else -1.0
            variant["metadata"]["best_canonical_y"] = canonical_peaks[0][1] if canonical_peaks else -1.0
            variant["metadata"]["best_score"] = canonical_peaks[0][2] if canonical_peaks else -999.0
            
            family_results.append({
                "metadata": variant["metadata"],
                "canonical_peaks": canonical_peaks,
                "res_hybrid": res_hybrid
            })
                
        # Phase 2 Enhancements
        
        # A. Candidate Aggregation
        clusters = cluster_candidates(all_peaks, support_radius=self.support_radius)
        
        # B. Family Voting
        clusters = calculate_family_support(clusters, total_families=len(variants))
        
        # C. Lattice-Aware NMS
        top_clusters, lattice_pairs = lattice_aware_nms(clusters, nms_radius=self.diagnostic_radius, top_k=self.top_k)
        
        # D. Peak/Runner-up Confidence
        classification, margin, ratio = classify_ambiguity(top_clusters)
        
        # E. Structural Candidate Selection
        selected_cluster = select_candidate(top_clusters, search_img, ref_100, classification)
        
        return {
            "family_results": family_results,
            "candidate_clusters": top_clusters,
            "lattice_pairs": lattice_pairs,
            "classification": classification,
            "margin": margin,
            "ratio": ratio,
            "selected_cluster": selected_cluster,
            "runtime": time.time() - t0,
            "search_matched": search_matched,
            "search_img": search_img
        }
