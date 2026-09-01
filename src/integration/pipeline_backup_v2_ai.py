import time
import os

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

    def __init__(self, top_k: int = 20, nms_radius: int = None):

        # ------------------------------------------------------------
        # PHASE 1: IMAGE CONDITIONING
        # ------------------------------------------------------------

        self.ice = ImageConditioningEngine()

        # ------------------------------------------------------------
        # PHASE 1 / GSPE
        # ------------------------------------------------------------

        scales = [
            9.0,
            9.5,
            10.0,
            10.5,
            11.0
        ]

        rotations = [
            -5.0,
            -2.5,
            0.0,
            2.5,
            5.0
        ]

        self.gspe = GlobalSearchProposalEngine(
            top_k=top_k,
            nms_radius=nms_radius,
            scale_hypotheses=scales,
            rotation_hypotheses=rotations
        )

        # ------------------------------------------------------------
        # Remaining Phase 1 modules
        # ------------------------------------------------------------

        self.gfee = GeometricFeatureExtractionEngine()
        self.srae = SpatialRegistrationAlignmentEngine()
        self.loc = ClassicalLocalization()

        # ------------------------------------------------------------
        # PHASE 2: SNRN
        # ------------------------------------------------------------

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.snrn = SNRN().to(self.device)

        # ------------------------------------------------------------
        # Load trained SNRN checkpoint
        # ------------------------------------------------------------

        project_root = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                ".."
            )
        )

        checkpoint_path_models = os.path.join(
            project_root,
            "models",
            "best_model.pth"
        )

        checkpoint_path_outputs = os.path.join(
            "outputs",
            "checkpoints",
            "best_model.pth"
        )

        if os.path.exists(checkpoint_path_models):
            checkpoint_path = checkpoint_path_models
        else:
            checkpoint_path = checkpoint_path_outputs

        if os.path.exists(checkpoint_path):

            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device
            )

            if (
                isinstance(checkpoint, dict)
                and "model_state_dict" in checkpoint
            ):
                self.snrn.load_state_dict(
                    checkpoint["model_state_dict"]
                )
            else:
                self.snrn.load_state_dict(checkpoint)

            print(
                f"\n[INFO] Successfully loaded AI refinement weights "
                f"from {checkpoint_path}"
            )

        else:

            print(
                "\n[WARNING] SNRN checkpoint not found. "
                "Using initialized network weights."
            )

        self.snrn.eval()

        # ------------------------------------------------------------
        # PHASE 3: DECISION FUSION
        # ------------------------------------------------------------

        self.fusion = DecisionFusionEngine()

        # ------------------------------------------------------------
        # Runtime state
        # ------------------------------------------------------------

        self.state = {}

    # =================================================================
    # STEP 36G
    #
    # Candidate-family analysis.
    #
    # IMPORTANT:
    # This is intentionally conservative.
    #
    # We do NOT blindly replace GSPE Top-1.
    #
    # The selector:
    #   1. Keeps every GSPE candidate.
    #   2. Measures spatial separation.
    #   3. Detects near-tied candidate families.
    #   4. Uses score margins to decide whether Top-1 is trustworthy.
    #   5. Only permits an alternative candidate when its score
    #      advantage is sufficiently meaningful.
    #
    # This protects the baseline from periodic aliases.
    # =================================================================

    @staticmethod
    def _candidate_center(box):

        return np.array(
            [
                float(box[0]) + float(box[2]) / 2.0,
                float(box[1]) + float(box[3]) / 2.0
            ],
            dtype=np.float32
        )

    @staticmethod
    def _candidate_distance(box_a, box_b):

        ca = HybridNavigationPipeline._candidate_center(box_a)
        cb = HybridNavigationPipeline._candidate_center(box_b)

        return float(np.linalg.norm(ca - cb))

    def _analyze_candidate_family(self, boxes, scores):

        candidates = []

        count = min(
            len(boxes),
            len(scores)
        )

        for i in range(count):

            box = boxes[i]

            center = self._candidate_center(box)

            candidates.append(
                {
                    "rank": i + 1,
                    "box": box,
                    "score": float(scores[i]),
                    "center_x": float(center[0]),
                    "center_y": float(center[1]),
                    "scale": float(box[4]),
                    "rotation": float(box[5]),
                }
            )

        if not candidates:
            return {
                "candidates": [],
                "family": [],
                "family_size": 0,
                "selected_rank": 1,
                "selection_reason": "NO_CANDIDATES"
            }

        top_score = candidates[0]["score"]

        # ------------------------------------------------------------
        # Near-tie threshold
        #
        # Periodic aliases in the V4 dataset frequently have extremely
        # small score differences. We therefore explicitly identify
        # such candidates rather than pretending the numerical ranking
        # is decisive.
        # ------------------------------------------------------------

        near_tie_abs = 0.0015

        family = []

        for candidate in candidates:

            score_delta = (
                top_score
                - candidate["score"]
            )

            if score_delta <= near_tie_abs:
                family.append(candidate)

        # ------------------------------------------------------------
        # Determine spatial separation from Top-1.
        #
        # Candidates very close to each other are generally part of the
        # same local peak. Candidates hundreds of pixels apart are
        # independent spatial hypotheses and may represent periodic
        # aliases.
        # ------------------------------------------------------------

        for candidate in candidates:

            candidate["distance_from_top1"] = (
                self._candidate_distance(
                    candidates[0]["box"],
                    candidate["box"]
                )
            )

        # ------------------------------------------------------------
        # Count genuinely separated members of the near-tied family.
        # ------------------------------------------------------------

        separated_family = []

        spatial_separation_threshold = 50.0

        for candidate in family:

            if (
                candidate["distance_from_top1"]
                >= spatial_separation_threshold
            ):
                separated_family.append(candidate)

        # Top-1 is always a member of the effective family.
        effective_family_size = 1 + len(
            separated_family
        )

        # ------------------------------------------------------------
        # Conservative selection policy
        #
        # For now, Top-1 remains the production choice unless another
        # candidate has a clear score advantage over Top-1.
        #
        # Since candidates are already sorted by GSPE score, an
        # alternative candidate cannot normally have a higher score.
        #
        # Therefore this stage primarily provides trustworthy family
        # diagnostics without introducing an uncontrolled regression.
        # ------------------------------------------------------------

        selected_rank = 1
        selection_reason = "TOP1_SCORE_MAX"

        if effective_family_size > 1:

            selection_reason = (
                "PERIODIC_FAMILY_TOP1_RETAINED"
            )

        return {
            "candidates": candidates,
            "family": family,
            "separated_family": separated_family,
            "family_size": effective_family_size,
            "selected_rank": selected_rank,
            "selection_reason": selection_reason
        }

    # =================================================================
    # RUN
    # =================================================================

    def run(self, reference, search):

        t0 = time.time()

        self.state = {
            "modules": {},
            "error": None
        }

        try:

            # ========================================================
            # PHASE 1: IMAGE CONDITIONING
            # ========================================================

            cond = self.ice.run(
                {
                    "reference": reference,
                    "search": search
                }
            )

            self.state["modules"]["ICE"] = True

            # ========================================================
            # PHASE 1: GSPE
            # ========================================================

            gspe_res = self.gspe.run(
                {
                    "reference": cond["reference_cond"],
                    "search": cond["search_cond"]
                }
            )

            self.state["modules"]["GSPE"] = True

            if gspe_res is None:
                raise RuntimeError(
                    "GSPE returned None."
                )

            boxes = gspe_res.get(
                "boxes",
                []
            )

            scores = gspe_res.get(
                "scores",
                []
            )

            if len(boxes) == 0:
                raise RuntimeError(
                    "GSPE returned no candidate boxes."
                )

            # ========================================================
            # STEP 36G: CANDIDATE-FAMILY ANALYSIS
            # ========================================================

            family_info = (
                self._analyze_candidate_family(
                    boxes,
                    scores
                )
            )

            candidates = family_info["candidates"]

            print(
                "\n========================================================"
            )
            print(
                "STEP 36G — GSPE CANDIDATE FAMILY ANALYSIS"
            )
            print(
                "========================================================"
            )

            print(
                f"Candidates available : "
                f"{len(candidates)}"
            )

            print(
                f"Near-tied family    : "
                f"{family_info['family_size']}"
            )

            print(
                f"Selection reason    : "
                f"{family_info['selection_reason']}"
            )

            for candidate in candidates:

                print(
                    f"[STEP36G] Candidate "
                    f"{candidate['rank']}: "
                    f"center=("
                    f"{candidate['center_x']:.3f}, "
                    f"{candidate['center_y']:.3f}), "
                    f"score="
                    f"{candidate['score']:.6f}, "
                    f"scale="
                    f"{candidate['scale']:.3f}, "
                    f"rot="
                    f"{candidate['rotation']:.3f}, "
                    f"distance="
                    f"{candidate.get('distance_from_top1', 0.0):.3f}"
                )

            print(
                "========================================================"
            )

            # ========================================================
            # Select production candidate
            #
            # Step 36G is deliberately conservative:
            # candidate 1 remains the production candidate unless
            # future experiments establish a validated replacement
            # rule.
            # ========================================================

            selected_rank = (
                family_info["selected_rank"]
            )

            selected_index = (
                selected_rank - 1
            )

            box = boxes[selected_index]

            top1_ncc = (
                float(scores[selected_index])
                if len(scores) > selected_index
                else 0.0
            )

            x = float(box[0])
            y = float(box[1])
            w = int(box[2])
            h = int(box[3])

            # --------------------------------------------------------
            # Extract candidate crop
            # --------------------------------------------------------

            search_cond = cond["search_cond"]

            cand_crop = search_cond[
                int(y):int(y + h),
                int(x):int(x + w)
            ]

            if cand_crop.size == 0:
                raise RuntimeError(
                    "GSPE candidate crop is empty."
                )

            center_x = (
                x + float(w) / 2.0
            )

            center_y = (
                y + float(h) / 2.0
            )

            classical_coord = np.array(
                [
                    center_x,
                    center_y
                ],
                dtype=np.float32
            )

            print(
                "\n--- GSPE TOP-1 SELECTED ---"
            )

            print(
                f"GSPE Coordinate : "
                f"({classical_coord[0]:.2f}, "
                f"{classical_coord[1]:.2f})"
            )

            print(
                f"GSPE Score      : "
                f"{top1_ncc:.4f}"
            )

            # ========================================================
            # SNRN PATCH PREPARATION
            # ========================================================

            def center_crop(img, size):

                h_img, w_img = img.shape

                ch = size
                cw = size

                y_start = max(
                    0,
                    h_img // 2 - ch // 2
                )

                x_start = max(
                    0,
                    w_img // 2 - cw // 2
                )

                crop = img[
                    y_start:y_start + ch,
                    x_start:x_start + cw
                ]

                if (
                    crop.shape[0] < ch
                    or crop.shape[1] < cw
                ):

                    crop = cv2.copyMakeBorder(
                        crop,
                        0,
                        ch - crop.shape[0],
                        0,
                        cw - crop.shape[1],
                        cv2.BORDER_CONSTANT,
                        value=0
                    )

                return crop

            # --------------------------------------------------------
            # Reference patch
            # --------------------------------------------------------

            ref_full = cond[
                "reference_cond"
            ]

            scale_ratio = 10.0

            w_ref = ref_full.shape[1]
            h_ref = ref_full.shape[0]

            scaled_w = int(
                round(
                    w_ref / scale_ratio
                )
            )

            scaled_h = int(
                round(
                    h_ref / scale_ratio
                )
            )

            ref_scaled = cv2.resize(
                ref_full,
                (
                    scaled_w,
                    scaled_h
                ),
                interpolation=cv2.INTER_AREA
            )

            ref_patch_np = (
                center_crop(
                    ref_scaled,
                    SNRNConfig.PATCH_SIZE
                )
                .astype(np.float32)
                / 255.0
            )

            ref_tensor = (
                torch.from_numpy(
                    ref_patch_np
                )
                .unsqueeze(0)
                .unsqueeze(0)
                .to(self.device)
            )

            # --------------------------------------------------------
            # Candidate patch
            # --------------------------------------------------------

            local_class_x = (
                classical_coord[0]
                - x
            )

            local_class_y = (
                classical_coord[1]
                - y
            )

            M_extract = np.array(
                [
                    [
                        1.0,
                        0.0,
                        (
                            SNRNConfig.PATCH_SIZE / 2.0
                            - local_class_x
                        )
                    ],
                    [
                        0.0,
                        1.0,
                        (
                            SNRNConfig.PATCH_SIZE / 2.0
                            - local_class_y
                        )
                    ]
                ],
                dtype=np.float32
            )

            cand_patch_raw = cv2.warpAffine(
                cand_crop,
                M_extract,
                (
                    SNRNConfig.PATCH_SIZE,
                    SNRNConfig.PATCH_SIZE
                ),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT
            )

            cand_patch_np = (
                cand_patch_raw
                .astype(np.float32)
                / 255.0
            )

            cand_tensor = (
                torch.from_numpy(
                    cand_patch_np
                )
                .unsqueeze(0)
                .unsqueeze(0)
                .to(self.device)
            )

            # ========================================================
            # PHASE 2: SNRN INFERENCE
            # ========================================================

            with torch.no_grad():

                ai_preds = self.snrn(
                    ref_tensor,
                    cand_tensor
                )

            self.state["modules"]["GFEE"] = False
            self.state["modules"]["SRAE"] = False
            self.state["modules"]["Localization"] = False
            self.state["modules"]["AI Refinement"] = True

            self.state["coarse_box"] = box
            self.state["reference_patch"] = ref_patch_np
            self.state["rectified_patch"] = cand_patch_np

            # ========================================================
            # Read SNRN outputs
            # ========================================================

            res_dx = float(
                ai_preds["residual"][0, 0]
                .detach()
                .cpu()
                .item()
            )

            res_dy = float(
                ai_preds["residual"][0, 1]
                .detach()
                .cpu()
                .item()
            )

            confidence = float(
                ai_preds["confidence"][0]
                .detach()
                .cpu()
                .item()
            )

            res_mag = float(
                np.sqrt(
                    res_dx ** 2
                    + res_dy ** 2
                )
            )

            # --------------------------------------------------------
            # Numerical validation
            # --------------------------------------------------------

            if (
                np.isnan(res_dx)
                or np.isnan(res_dy)
                or np.isinf(res_dx)
                or np.isinf(res_dy)
            ):

                raise ValueError(
                    "AI predicted invalid residual coordinates: "
                    f"({res_dx}, {res_dy})"
                )

            if (
                np.isnan(confidence)
                or np.isinf(confidence)
            ):

                raise ValueError(
                    "AI predicted invalid confidence: "
                    f"{confidence}"
                )

            print(
                "\n--- AI PREDICTION ---"
            )

            print(
                f"Residual dx        : "
                f"{res_dx:.4f} px"
            )

            print(
                f"Residual dy        : "
                f"{res_dy:.4f} px"
            )

            print(
                f"Residual Magnitude : "
                f"{res_mag:.4f} px"
            )

            print(
                f"Confidence Score   : "
                f"{confidence:.4f}"
            )

            print(
                "---------------------\n"
            )

            # ========================================================
            # PHASE 3: DECISION FUSION
            # ========================================================

            max_allowed_res = getattr(
                SNRNConfig,
                "MAX_CLASSICAL_ERROR",
                5.0
            )

            conf_threshold = 0.90

            ai_coord = np.array(
                [
                    classical_coord[0]
                    + res_dx,
                    classical_coord[1]
                    + res_dy
                ],
                dtype=np.float32
            )

            print(
                "\n--- DECISION FUSION ---"
            )

            print(
                f"Classical : "
                f"[{classical_coord[0]:.4f}, "
                f"{classical_coord[1]:.4f}]"
            )

            print(
                f"AI        : "
                f"[{ai_coord[0]:.4f}, "
                f"{ai_coord[1]:.4f}]"
            )

            print(
                f"Residual  : "
                f"[{res_dx:.4f}, "
                f"{res_dy:.4f}]"
            )

            print(
                f"Magnitude : "
                f"{res_mag:.4f} px"
            )

            print(
                f"Confidence: "
                f"{confidence:.4f}"
            )

            print(
                f"Max Allow : "
                f"{max_allowed_res:.4f} px"
            )

            if (
                res_mag <= max_allowed_res
                and confidence >= conf_threshold
            ):

                final_coord = ai_coord
                decision = "AI_REFINED"

                self.state["modules"]["Fusion"] = True

            else:

                final_coord = classical_coord
                decision = (
                    "CLASSICAL_GSPE_FALLBACK"
                )

                self.state["modules"]["Fusion"] = False

            print(
                f"Decision  : {decision}"
            )

            # ========================================================
            # FINAL STATE
            # ========================================================

            dist_from_gspe = float(
                np.linalg.norm(
                    final_coord
                    - classical_coord
                )
            )

            # --------------------------------------------------------
            # GSPE state
            # --------------------------------------------------------

            self.state[
                "gspe_selected_rank"
            ] = selected_rank

            self.state[
                "gspe_selected_score"
            ] = top1_ncc

            self.state[
                "gspe_candidate_count"
            ] = len(candidates)

            self.state[
                "gspe_family_size"
            ] = family_info[
                "family_size"
            ]

            self.state[
                "gspe_family_selection_reason"
            ] = family_info[
                "selection_reason"
            ]

            self.state[
                "gspe_candidates"
            ] = candidates

            self.state[
                "gspe_family"
            ] = family_info[
                "family"
            ]

            self.state[
                "gspe_separated_family"
            ] = family_info[
                "separated_family"
            ]

            self.state[
                "gspe_periodic_family_detected"
            ] = (
                family_info["family_size"] > 1
            )

            # --------------------------------------------------------
            # Coordinates
            # --------------------------------------------------------

            self.state[
                "classical_coord"
            ] = classical_coord

            self.state[
                "ai_coord"
            ] = ai_coord

            self.state[
                "final_coord"
            ] = final_coord

            self.state[
                "confidence"
            ] = confidence

            self.state[
                "decision"
            ] = decision

            self.state[
                "dist_from_gspe"
            ] = dist_from_gspe

            # --------------------------------------------------------
            # AI residual
            # --------------------------------------------------------

            self.state[
                "ai_residual"
            ] = np.array(
                [
                    res_dx,
                    res_dy
                ],
                dtype=np.float32
            )

            self.state[
                "ai_residual_mag"
            ] = res_mag

            # --------------------------------------------------------
            # Runtime
            # --------------------------------------------------------

            self.state[
                "runtime"
            ] = time.time() - t0

            return self.state

        except Exception as e:

            self.state[
                "error"
            ] = str(e)

            self.state[
                "runtime"
            ] = time.time() - t0

            return self.state
