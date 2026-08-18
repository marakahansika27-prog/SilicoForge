import os
import sys
import cv2
import numpy as np

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )
)

from src.integration.pipeline import HybridNavigationPipeline
from src.coarse_search.gspe import GlobalSearchProposalEngine
from src.feature_matching.gfee import GeometricFeatureExtractionEngine
from src.registration.srae import SpatialRegistrationAlignmentEngine
from scripts.benchmark_40_cases import generate_benchmark_case


def calculate_distance(x1, y1, x2, y2):
    return np.sqrt(
        (x1 - x2) ** 2 +
        (y1 - y2) ** 2
    )


def main():

    print("========================================")
    print("FINAL RANKING & PERIODICITY DIAGNOSTIC")
    print("========================================")

    # ==========================================================
    # PIPELINE / DIAGNOSTIC CONFIGURATION
    # ==========================================================

    pipeline = HybridNavigationPipeline(
        top_k=20,
        nms_radius=50
    )

    gspe = GlobalSearchProposalEngine(
        top_k=20,
        nms_radius=50,
        scale_hypotheses=[10.0],
        rotation_hypotheses=[0.0]
    )

    gfee = GeometricFeatureExtractionEngine()

    srae = SpatialRegistrationAlignmentEngine()

    # ==========================================================
    # 40-CASE BENCHMARK DEFINITION
    # ==========================================================

    cases_def = []

    for i in range(8):
        cases_def.append(("DRAM", "easy"))

    for i in range(8):
        cases_def.append(("DRAM", "moderate"))

    for i in range(6):
        cases_def.append(("DRAM", "hard"))

    for i in range(8):
        cases_def.append(("FinFET", "easy"))

    for i in range(6):
        cases_def.append(("FinFET", "moderate"))

    for i in range(4):
        cases_def.append(("FinFET", "hard"))

    # ==========================================================
    # GLOBAL METRICS
    # ==========================================================

    ncc_wins = 0
    srae_wins = 0

    top3_ncc_hits = 0
    top5_ncc_hits = 0
    top10_ncc_hits = 0
    top20_ncc_hits = 0

    total_valid = 0

    # ==========================================================
    # TABLE HEADER
    # ==========================================================

    print(
        "\n| Case | Rank | X | Y | NCC Score | "
        "SRAE Inliers | Dist to GT | Selected |"
    )

    print(
        "|------|------|---|---|-----------|"
        "--------------|------------|----------|"
    )

    # ==========================================================
    # CASE LOOP
    # ==========================================================

    for idx, (arch, diff) in enumerate(cases_def):

        seed = 1000 + idx + 1

        print(
            f"\n\n========== CASE {idx + 1:02d}/40 "
            f"{arch} {diff} =========="
        )

        # ------------------------------------------------------
        # Generate benchmark case
        # ------------------------------------------------------

        ref_img, search_img, gt_x, gt_y = (
            generate_benchmark_case(
                seed,
                arch,
                diff
            )
        )

        print(
            f"Ground Truth : "
            f"({gt_x:.4f}, {gt_y:.4f})"
        )

        # ------------------------------------------------------
        # ICE
        # ------------------------------------------------------

        cond = pipeline.ice.run({
            'reference': ref_img,
            'search': search_img
        })

        # ------------------------------------------------------
        # GSPE
        # ------------------------------------------------------

        gspe_res = gspe.run({
            'reference': cond['reference_cond'],
            'search': cond['search_cond']
        })

        boxes = gspe_res.get(
            'boxes',
            []
        )

        scores = gspe_res.get(
            'scores',
            []
        )

        if not boxes:

            print(
                "No GSPE candidates found."
            )

            continue

        # ------------------------------------------------------
        # Candidate storage
        # ------------------------------------------------------

        candidates = []

        best_inliers = -1

        srae_winner_idx = -1

        # ======================================================
        # EVALUATE ALL GSPE CANDIDATES
        # ======================================================

        for rank_idx, (box, score) in enumerate(
            zip(boxes, scores)
        ):

            # --------------------------------------------------
            # GSPE box format:
            #
            # x, y, w, h, scale, rotation
            #
            # We only need first four values for cropping.
            # --------------------------------------------------

            x, y, w, h = box[:4]

            # --------------------------------------------------
            # Subpixel center
            # --------------------------------------------------

            center_x = (
                float(x) +
                float(w) / 2.0
            )

            center_y = (
                float(y) +
                float(h) / 2.0
            )

            # --------------------------------------------------
            # Distance from ground truth
            # --------------------------------------------------

            dist = calculate_distance(
                center_x,
                center_y,
                gt_x,
                gt_y
            )

            # --------------------------------------------------
            # Convert ONLY crop coordinates to integers
            #
            # The localization itself remains subpixel.
            # --------------------------------------------------

            x0 = int(
                round(float(x))
            )

            y0 = int(
                round(float(y))
            )

            w0 = max(
                1,
                int(round(float(w)))
            )

            h0 = max(
                1,
                int(round(float(h)))
            )

            # --------------------------------------------------
            # Clamp crop to image boundaries
            # --------------------------------------------------

            img_h, img_w = (
                cond['search_cond'].shape[:2]
            )

            x0 = max(
                0,
                min(x0, img_w - 1)
            )

            y0 = max(
                0,
                min(y0, img_h - 1)
            )

            x1 = min(
                img_w,
                x0 + w0
            )

            y1 = min(
                img_h,
                y0 + h0
            )

            cand_crop = cond[
                'search_cond'
            ][
                y0:y1,
                x0:x1
            ]

            # --------------------------------------------------
            # Guard against empty crops
            # --------------------------------------------------

            if (
                cand_crop is None
                or cand_crop.size == 0
            ):

                inliers = 0

                candidates.append({
                    'idx': rank_idx,
                    'x': center_x,
                    'y': center_y,
                    'ncc': float(score),
                    'inliers': inliers,
                    'dist': dist
                })

                continue

            # ==================================================
            # GFEE
            # ==================================================

            try:

                gfee_res = gfee.run({
                    'reference':
                        cond['reference_cond'],

                    'candidate':
                        cand_crop
                })

            except Exception as exc:

                print(
                    f"GFEE warning at candidate "
                    f"{rank_idx + 1}: {exc}"
                )

                gfee_res = {}

            # ==================================================
            # SRAE
            # ==================================================

            try:

                srae_res = srae.run({
                    'reference':
                        cond['reference_cond'],

                    'candidate':
                        cand_crop,

                    'kp1':
                        gfee_res.get(
                            'kp1',
                            []
                        ),

                    'kp2':
                        gfee_res.get(
                            'kp2',
                            []
                        ),

                    'matches':
                        gfee_res.get(
                            'good_matches',
                            []
                        )
                })

            except Exception as exc:

                print(
                    f"SRAE warning at candidate "
                    f"{rank_idx + 1}: {exc}"
                )

                srae_res = {
                    'stats': {
                        'inliers': 0
                    }
                }

            # ==================================================
            # READ SRAE INLIERS SAFELY
            # ==================================================

            stats = srae_res.get(
                'stats',
                {}
            )

            if stats is None:
                stats = {}

            inliers = stats.get(
                'inliers',
                0
            )

            try:
                inliers = int(
                    inliers
                )
            except Exception:
                inliers = 0

            # ==================================================
            # STORE CANDIDATE
            # ==================================================

            candidates.append({
                'idx': rank_idx,
                'x': center_x,
                'y': center_y,
                'ncc': float(score),
                'inliers': inliers,
                'dist': float(dist)
            })

            # ==================================================
            # TRACK SRAE WINNER
            # ==================================================

            if inliers > best_inliers:

                best_inliers = inliers

                srae_winner_idx = (
                    rank_idx
                )

        # ======================================================
        # FIND TRUE CANDIDATE
        # ======================================================

        if not candidates:
            continue

        true_cand = min(
            candidates,
            key=lambda c: c['dist']
        )

        # ------------------------------------------------------
        # Candidate considered present if within 25 px
        # ------------------------------------------------------

        is_present = (
            true_cand['dist'] <= 25.0
        )

        if not is_present:

            print(
                f"TRUE CANDIDATE NOT IN "
                f"25px REGION | "
                f"Closest distance = "
                f"{true_cand['dist']:.2f}px"
            )

            continue

        total_valid += 1

        # ======================================================
        # NCC RANKING
        # ======================================================

        ncc_sorted = sorted(
            candidates,
            key=lambda c: c['ncc'],
            reverse=True
        )

        ncc_rank = (
            next(
                i
                for i, c in enumerate(
                    ncc_sorted
                )
                if c['idx'] ==
                true_cand['idx']
            )
            + 1
        )

        # ======================================================
        # SRAE RANKING
        # ======================================================

        srae_sorted = sorted(
            candidates,
            key=lambda c: c['inliers'],
            reverse=True
        )

        srae_rank = (
            next(
                i
                for i, c in enumerate(
                    srae_sorted
                )
                if c['idx'] ==
                true_cand['idx']
            )
            + 1
        )

        # ======================================================
        # METRICS
        # ======================================================

        if ncc_rank == 1:
            ncc_wins += 1

        if srae_rank == 1:
            srae_wins += 1

        if ncc_rank <= 3:
            top3_ncc_hits += 1

        if ncc_rank <= 5:
            top5_ncc_hits += 1

        if ncc_rank <= 10:
            top10_ncc_hits += 1

        if ncc_rank <= 20:
            top20_ncc_hits += 1

        # ======================================================
        # CASE SUMMARY
        # ======================================================

        selected_candidate = None

        if (
            srae_winner_idx >= 0
            and
            srae_winner_idx < len(candidates)
        ):

            selected_candidate = candidates[
                srae_winner_idx
            ]

        print(
            f"\nTrue Candidate:"
        )

        print(
            f"  Rank      : {true_cand['idx'] + 1}"
        )

        print(
            f"  Position  : "
            f"({true_cand['x']:.4f}, "
            f"{true_cand['y']:.4f})"
        )

        print(
            f"  NCC       : "
            f"{true_cand['ncc']:.6f}"
        )

        print(
            f"  SRAE      : "
            f"{true_cand['inliers']}"
        )

        print(
            f"  Distance  : "
            f"{true_cand['dist']:.4f}px"
        )

        print(
            f"  NCC Rank  : {ncc_rank}"
        )

        print(
            f"  SRAE Rank : {srae_rank}"
        )

        if selected_candidate is not None:

            print(
                f"  Selected  : "
                f"Rank {selected_candidate['idx'] + 1}"
            )

        # ======================================================
        # DETAILED FAILURE CASES
        # ======================================================

        if idx in [
            4,
            11,
            13,
            15,
            17
        ]:

            print(
                "\n--- DETAILED CANDIDATES ---"
            )

            for c in candidates:

                selected = (
                    "YES"
                    if c['idx'] ==
                    srae_winner_idx
                    else ""
                )

                print(
                    f"| {idx + 1:02d} | "
                    f"{c['idx'] + 1:02d} | "
                    f"{c['x']:.2f} | "
                    f"{c['y']:.2f} | "
                    f"{c['ncc']:.6f} | "
                    f"{c['inliers']:3d} | "
                    f"{c['dist']:7.2f} | "
                    f"{selected:3s} |"
                )

            # --------------------------------------------------
            # Periodicity analysis
            # --------------------------------------------------

            if len(candidates) >= 2:

                d01 = calculate_distance(
                    candidates[0]['x'],
                    candidates[0]['y'],
                    candidates[1]['x'],
                    candidates[1]['y']
                )

                print(
                    f"\n-> PERIODICITY:"
                )

                print(
                    f"Peak 1 and Peak 2 "
                    f"separation = "
                    f"{d01:.2f}px"
                )

                print(
                    "Expected DRAM pitch: "
                    "100px"
                )

                if (
                    0 <= srae_winner_idx
                    < len(candidates)
                ):

                    selected_cand = (
                        candidates[
                            srae_winner_idx
                        ]
                    )

                    print(
                        f"-> True NCC vs "
                        f"Selected NCC difference: "
                        f"{true_cand['ncc'] - selected_cand['ncc']:.6f}"
                    )

                    print(
                        f"-> True SRAE vs "
                        f"Selected SRAE difference: "
                        f"{true_cand['inliers'] - selected_cand['inliers']}"
                    )

            print(
                "---------------------------------------------------------"
            )

    # ==========================================================
    # FINAL SUMMARY
    # ==========================================================

    print(
        "\n========================================"
    )

    print(
        "RANKING PERFORMANCE SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        f"Total Valid Cases "
        f"(True Target in Top-20): "
        f"{total_valid}/40"
    )

    if total_valid > 0:

        print(
            f"NCC ranks True Candidate #1: "
            f"{ncc_wins / total_valid * 100:.1f}%"
        )

        print(
            f"SRAE ranks True Candidate #1: "
            f"{srae_wins / total_valid * 100:.1f}%"
        )

        print(
            f"Top-3 NCC Recall: "
            f"{top3_ncc_hits / total_valid * 100:.1f}%"
        )

        print(
            f"Top-5 NCC Recall: "
            f"{top5_ncc_hits / total_valid * 100:.1f}%"
        )

        print(
            f"Top-10 NCC Recall: "
            f"{top10_ncc_hits / total_valid * 100:.1f}%"
        )

        print(
            f"Top-20 NCC Recall: "
            f"{top20_ncc_hits / total_valid * 100:.1f}%"
        )

    else:

        print(
            "No valid cases were found "
            "within the 25px candidate threshold."
        )


if __name__ == "__main__":
    main()