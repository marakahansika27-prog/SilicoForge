import time
import torch
import cv2
import numpy as np
import io
import contextlib

# ============================================================
# PIPELINE COMPONENTS
# ============================================================

from src.preprocessing.ice import ImageConditioningEngine
from src.coarse_search.gspe import GlobalSearchProposalEngine
from src.feature_matching.gfee import GeometricFeatureExtractionEngine
from src.registration.srae import SpatialRegistrationAlignmentEngine
from src.localization.localization import ClassicalLocalization
from src.integration.decision_fusion import DecisionFusionEngine


# ============================================================
# SUBPIXEL REFINEMENT
# ============================================================

def get_subpixel(res, x, y):
    """
    Estimate subpixel peak location using quadratic interpolation.

    Returns:
        (subpixel_x, subpixel_y)
    """

    h, w = res.shape

    dx = 0.0
    dy = 0.0

    # --------------------------------------------------------
    # X-direction interpolation
    # --------------------------------------------------------

    if 0 < x < w - 1 and 0 < y < h - 1:

        fx_m1 = float(res[y, x - 1])
        fx_0 = float(res[y, x])
        fx_p1 = float(res[y, x + 1])

        denom_x = (
            fx_m1
            - 2.0 * fx_0
            + fx_p1
        )

        if abs(denom_x) > 1e-6:

            val_x = (
                0.5
                * (fx_m1 - fx_p1)
                / denom_x
            )

            if (
                not np.isnan(val_x)
                and not np.isinf(val_x)
                and abs(val_x) <= 0.5
            ):
                dx = val_x

        # ----------------------------------------------------
        # Y-direction interpolation
        # ----------------------------------------------------

        fy_m1 = float(res[y - 1, x])
        fy_0 = float(res[y, x])
        fy_p1 = float(res[y + 1, x])

        denom_y = (
            fy_m1
            - 2.0 * fy_0
            + fy_p1
        )

        if abs(denom_y) > 1e-6:

            val_y = (
                0.5
                * (fy_m1 - fy_p1)
                / denom_y
            )

            if (
                not np.isnan(val_y)
                and not np.isinf(val_y)
                and abs(val_y) <= 0.5
            ):
                dy = val_y

    return (
        float(x + dx),
        float(y + dy)
    )


# ============================================================
# LOCAL REFINEMENT
# ============================================================

def local_refine(res, cx, cy, radius):
    """
    Search for the strongest correlation peak inside a
    local neighborhood around (cx, cy).
    """

    h, w = res.shape

    y1 = max(
        0,
        int(cy - radius)
    )

    y2 = min(
        h,
        int(cy + radius + 1)
    )

    x1 = max(
        0,
        int(cx - radius)
    )

    x2 = min(
        w,
        int(cx + radius + 1)
    )

    if y2 <= y1 or x2 <= x1:
        return (
            float(cx),
            float(cy)
        )

    window = res[
        y1:y2,
        x1:x2
    ]

    _, _, _, max_loc = cv2.minMaxLoc(
        window
    )

    px = x1 + max_loc[0]
    py = y1 + max_loc[1]

    return (
        float(px),
        float(py)
    )


# ============================================================
# LOCAL SUBPIXEL REFINEMENT
# ============================================================

def local_refine_subpixel(
    res,
    cx,
    cy,
    radius
):
    """
    First performs local integer refinement,
    then applies subpixel quadratic interpolation.
    """

    px, py = local_refine(
        res,
        cx,
        cy,
        radius
    )

    return get_subpixel(
        res,
        int(round(px)),
        int(round(py))
    )


# ============================================================
# HYBRID NAVIGATION PIPELINE
# ============================================================

class HybridNavigationPipeline:

    def __init__(
        self,
        top_k: int = 1,
        nms_radius: int = None
    ):
        """
        Frozen Phase 28 production configuration.

        Scale    : 10.0
        Rotation : 0.0°
        Top-K    : configurable, production uses 1

        NOTE:
        Top-K macro reranking is intentionally NOT used.
        """

        # ----------------------------------------------------
        # Image Conditioning
        # ----------------------------------------------------

        self.ice = ImageConditioningEngine()

        # ----------------------------------------------------
        # GSPE
        # ----------------------------------------------------
        #
        # Phase 27/28 validation established:
        # scale = 10.0
        # rotation = 0.0
        # ----------------------------------------------------

        scales = [10.0]
        rotations = [0.0]

        self.gspe = GlobalSearchProposalEngine(
            top_k=top_k,
            nms_radius=nms_radius,
            scale_hypotheses=scales,
            rotation_hypotheses=rotations
        )

        # ----------------------------------------------------
        # Existing architecture components
        # ----------------------------------------------------

        self.gfee = (
            GeometricFeatureExtractionEngine()
        )

        self.srae = (
            SpatialRegistrationAlignmentEngine()
        )

        self.loc = ClassicalLocalization()

        # ----------------------------------------------------
        # Decision Fusion
        # ----------------------------------------------------

        self.fusion = DecisionFusionEngine(
            confidence_threshold=0.90,
            residual_deadband=0.10
        )

        self.state = {}


    # ========================================================
    # MAIN PIPELINE
    # ========================================================

    def run(
        self,
        ref_img,
        search_img,
        ref_macro=None
    ):
        """
        Executes the frozen Phase 28 classical pipeline.

        Pipeline:

            Reference + Search
                    ↓
                  ICE
                    ↓
             100px Local GSPE
                    ↓
             4000px Macro GSPE
                    ↓
             Best Macro Peak
                    ↓
             50px Local Refinement
                    ↓
             40px Subpixel Refinement
                    ↓
               Final (X,Y)

        IMPORTANT:
        No Top-K macro candidate reranking is performed.
        """

        t0 = time.time()

        self.state = {
            "reference": ref_img,
            "search": search_img,
            "modules": {}
        }

        try:

            # =================================================
            # 1. IMAGE CONDITIONING — ICE
            # =================================================

            cond = self.ice.run({
                "reference": ref_img,
                "search": search_img
            })

            self.state["modules"]["ICE"] = True

            ref_cond = (
                cond["reference_cond"]
            )

            search_cond = (
                cond["search_cond"]
            )


            # =================================================
            # 2. LOCAL 100px GSPE
            # =================================================

            with contextlib.redirect_stdout(io.StringIO()):
                gspe_res_100 = self.gspe.run({
                    "reference": ref_cond,
                    "search": search_cond
                })

            res_hybrid_100 = (
                gspe_res_100["res_hybrid"]
            )

            res_raw_100 = (
                gspe_res_100["res_raw"]
            )

            self.state["modules"]["GSPE_100"] = True


            # =================================================
            # 3. 4000px MACRO GSPE
            # =================================================
            #
            # IMPORTANT:
            # We deliberately use the strongest macro peak.
            #
            # The previously tested Top-K local-score
            # reranking was rejected because it reduced:
            #
            # Frozen:
            #   96.17 px mean
            #   30.0% @ 10px
            #
            # Top-K reranking:
            #   100.44 px mean
            #   21.7% @ 10px
            #
            # Therefore no candidate reranking here.
            # =================================================

            if ref_macro is not None:

                with contextlib.redirect_stdout(io.StringIO()):
                    gspe_res_macro = self.gspe.run({
                        "reference": ref_macro,
                        "search": search_cond
                    })

                self.state["modules"][
                    "GSPE_MACRO"
                ] = True

                (
                    _,
                    macro_score,
                    _,
                    max_loc_macro
                ) = cv2.minMaxLoc(
                    gspe_res_macro[
                        "res_hybrid"
                    ]
                )

                macro_cx = float(
                    max_loc_macro[0]
                )

                macro_cy = float(
                    max_loc_macro[1]
                )

                self.state[
                    "macro_coord"
                ] = np.array(
                    [
                        macro_cx,
                        macro_cy
                    ],
                    dtype=np.float32
                )

                self.state[
                    "macro_score"
                ] = float(macro_score)

                print(
                    "\n--- MACRO LOCALIZATION ---"
                )

                print(
                    "Macro Peak : "
                    f"({macro_cx:.4f}, "
                    f"{macro_cy:.4f})"
                )

                print(
                    "Macro Score: "
                    f"{float(macro_score):.4f}"
                )

            else:

                # =============================================
                # FALLBACK TO LOCAL GSPE
                # =============================================

                (
                    _,
                    local_score,
                    _,
                    max_loc_100
                ) = cv2.minMaxLoc(
                    res_hybrid_100
                )

                macro_cx = float(
                    max_loc_100[0]
                )

                macro_cy = float(
                    max_loc_100[1]
                )

                self.state["modules"][
                    "GSPE_MACRO"
                ] = False

                self.state[
                    "macro_coord"
                ] = np.array(
                    [
                        macro_cx,
                        macro_cy
                    ],
                    dtype=np.float32
                )

                self.state[
                    "macro_score"
                ] = float(local_score)

                print(
                    "\n--- MACRO FALLBACK ---"
                )

                print(
                    "Local Peak : "
                    f"({macro_cx:.4f}, "
                    f"{macro_cy:.4f})"
                )

                print(
                    "Local Score: "
                    f"{float(local_score):.4f}"
                )


            # =================================================
            # 4. 50px LOCAL REFINEMENT
            # =================================================

            loc_cx, loc_cy = local_refine(
                res_hybrid_100,
                macro_cx,
                macro_cy,
                50
            )

            self.state[
                "local_coord"
            ] = np.array(
                [
                    loc_cx,
                    loc_cy
                ],
                dtype=np.float32
            )

            print(
                "\n--- LOCAL REFINEMENT ---"
            )

            print(
                "Macro Coordinate : "
                f"({macro_cx:.4f}, "
                f"{macro_cy:.4f})"
            )

            print(
                "Local Coordinate : "
                f"({loc_cx:.4f}, "
                f"{loc_cy:.4f})"
            )


            # =================================================
            # 5. 40px SUBPIXEL REFINEMENT
            # =================================================

            final_x, final_y = (
                local_refine_subpixel(
                    res_raw_100,
                    loc_cx,
                    loc_cy,
                    40
                )
            )

            final_coord = np.array(
                [
                    final_x,
                    final_y
                ],
                dtype=np.float32
            )

            self.state[
                "subpixel_coord"
            ] = final_coord


            # =================================================
            # 6. MODULE STATUS
            # =================================================

            self.state[
                "modules"
            ]["GFEE"] = False

            self.state[
                "modules"
            ]["SRAE"] = False

            self.state[
                "modules"
            ]["Localization"] = False

            self.state[
                "modules"
            ]["AI Refinement"] = False

            self.state[
                "modules"
            ]["Fusion"] = True


            # =================================================
            # 7. FINAL OUTPUT
            # =================================================

            self.state[
                "final_coord"
            ] = final_coord

            self.state[
                "decision"
            ] = "MACRO_GSPE_FROZEN"

            self.state[
                "runtime"
            ] = time.time() - t0

            print(
                "\n--- FINAL LOCALIZATION ---"
            )

            print(
                "Final Coordinate : "
                f"({float(final_x):.4f}, "
                f"{float(final_y):.4f})"
            )

            print(
                "Runtime          : "
                f"{self.state['runtime'] * 1000:.2f} ms"
            )

            return self.state


        # =====================================================
        # ERROR HANDLING
        # =====================================================

        except Exception as exc:

            self.state[
                "error"
            ] = str(exc)

            self.state[
                "runtime"
            ] = time.time() - t0

            print(
                f"\nPIPELINE ERROR: {exc}"
            )

            return self.state