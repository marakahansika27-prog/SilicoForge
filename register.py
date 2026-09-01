import argparse
import csv
from pathlib import Path

import cv2


# ============================================================
# DRIFT-SENSE V2 — PHASE 2 REGISTRATION ENTRY POINT
# ============================================================

# Fixed rejection threshold calibrated on the local 200-case
# V4 benchmark.
#
# GSPE score >= 0.85  -> target PRESENT
# GSPE score <  0.85  -> target ABSENT / REJECT
#
# Do NOT use AI confidence for rejection.
GSPE_REJECTION_THRESHOLD = 0.85


def load_image(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Failed to load image: {path}")

    return image


def get_selected_candidate(result):
    """
    Retrieve the actual selected GSPE candidate.

    The pipeline exposes:
        gspe_candidates
        gspe_selected_rank

    Candidate geometry is stored inside the selected candidate.
    """

    candidates = result.get("gspe_candidates")

    if not candidates:
        return None

    try:
        selected_rank = int(
            result.get("gspe_selected_rank", 1)
        )
    except Exception:
        selected_rank = 1

    if selected_rank < 1:
        selected_rank = 1

    index = selected_rank - 1

    if index >= len(candidates):
        index = 0

    return candidates[index]


def extract_selected_geometry(result):
    """
    Extract scale and rotation from the actual selected GSPE
    candidate.

    Falls back to coarse_box if the candidate structure does
    not expose the values directly.
    """

    candidate = get_selected_candidate(result)

    scale = None
    rotation = None

    # --------------------------------------------------------
    # First choice: selected GSPE candidate
    # --------------------------------------------------------
    if isinstance(candidate, dict):

        # Scale
        for key in (
            "scale",
            "selected_scale",
            "best_scale",
        ):
            if key in candidate and candidate[key] is not None:
                try:
                    scale = float(candidate[key])
                    break
                except (TypeError, ValueError):
                    pass

        # Rotation
        for key in (
            "rotation",
            "rot",
            "theta",
            "selected_rotation",
            "best_rotation",
        ):
            if key in candidate and candidate[key] is not None:
                try:
                    rotation = float(candidate[key])
                    break
                except (TypeError, ValueError):
                    pass

    # --------------------------------------------------------
    # Second choice: coarse_box
    # --------------------------------------------------------
    coarse_box = result.get("coarse_box")

    if isinstance(coarse_box, dict):

        if scale is None:
            for key in (
                "scale",
                "selected_scale",
                "best_scale",
            ):
                if key in coarse_box and coarse_box[key] is not None:
                    try:
                        scale = float(coarse_box[key])
                        break
                    except (TypeError, ValueError):
                        pass

        if rotation is None:
            for key in (
                "rotation",
                "rot",
                "theta",
                "selected_rotation",
                "best_rotation",
            ):
                if key in coarse_box and coarse_box[key] is not None:
                    try:
                        rotation = float(coarse_box[key])
                        break
                    except (TypeError, ValueError):
                        pass

    # --------------------------------------------------------
    # Third choice: top-level result fields
    # --------------------------------------------------------
    if scale is None:
        for key in (
            "selected_scale",
            "scale",
        ):
            if key in result and result[key] is not None:
                try:
                    scale = float(result[key])
                    break
                except (TypeError, ValueError):
                    pass

    if rotation is None:
        for key in (
            "selected_rotation",
            "rotation",
            "theta",
        ):
            if key in result and result[key] is not None:
                try:
                    rotation = float(result[key])
                    break
                except (TypeError, ValueError):
                    pass

    # --------------------------------------------------------
    # Safe final fallback
    # --------------------------------------------------------
    if scale is None:
        scale = 0.0

    if rotation is None:
        rotation = 0.0

    return scale, rotation


def main():

    parser = argparse.ArgumentParser(
        description="Drift-Sense V2 Phase 2 registration entry point"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input pair CSV"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output predictions CSV"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_path}"
        )

    # --------------------------------------------------------
    # Import production pipeline
    # --------------------------------------------------------
    from src.integration.pipeline_backup_v2_ai import (
        HybridNavigationPipeline
    )

    pipeline = HybridNavigationPipeline()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # REQUIRED PHASE-2 OUTPUT CONTRACT
    # --------------------------------------------------------
    required_output_columns = [
        "pair_id",
        "x",
        "y",
        "theta",
        "scale",
        "found",
        "score",
    ]

    rows = []

    # --------------------------------------------------------
    # READ INPUT PAIRS
    # --------------------------------------------------------
    with input_path.open(
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(
                "Input CSV has no header."
            )

        field_map = {
            name.strip().lower(): name
            for name in reader.fieldnames
        }

        def find_column(*names):
            for name in names:
                if name.lower() in field_map:
                    return field_map[name.lower()]
            return None

        # ----------------------------------------------------
        # INPUT COLUMN DISCOVERY
        # ----------------------------------------------------
        pair_col = find_column(
            "pair_id",
            "case_id",
            "id"
        )

        ref_col = find_column(
            "reference",
            "reference_path",
            "reference_img",
            "reference_image",
            "ref"
        )

        search_col = find_column(
            "search",
            "search_path",
            "search_img",
            "search_image"
        )

        if pair_col is None:
            raise ValueError(
                "Input CSV must contain a pair_id column."
            )

        if ref_col is None or search_col is None:
            raise ValueError(
                "Input CSV must contain reference and "
                "search image path columns."
            )

        # ----------------------------------------------------
        # PROCESS EACH PAIR
        # ----------------------------------------------------
        for row in reader:

            pair_id = str(
                row.get(pair_col, "")
            ).strip()

            if not pair_id:
                continue

            try:

                # --------------------------------------------
                # LOAD IMAGES
                # --------------------------------------------
                ref_img = load_image(
                    row[ref_col]
                )

                search_img = load_image(
                    row[search_col]
                )

                # --------------------------------------------
                # RUN PRODUCTION PIPELINE
                # --------------------------------------------
                result = pipeline.run(
                    ref_img,
                    search_img
                )

                if result.get("error"):
                    raise RuntimeError(
                        str(result["error"])
                    )

                # --------------------------------------------
                # GET FINAL COORDINATE
                # --------------------------------------------
                coord = result.get(
                    "final_coord"
                )

                if coord is None:
                    raise RuntimeError(
                        "Pipeline did not return final_coord."
                    )

                if len(coord) < 2:
                    raise RuntimeError(
                        "Pipeline returned an invalid coordinate."
                    )

                x = float(coord[0])
                y = float(coord[1])

                # --------------------------------------------
                # GSPE SCORE
                # --------------------------------------------
                score_value = result.get(
                    "gspe_selected_score",
                    0.0
                )

                score = float(score_value)

                # --------------------------------------------
                # SELECTED GEOMETRY
                # --------------------------------------------
                scale, theta = extract_selected_geometry(
                    result
                )

                # --------------------------------------------
                # REJECTION DECISION
                # --------------------------------------------
                #
                # IMPORTANT:
                # Rejection is based ONLY on GSPE selected
                # score. AI confidence is intentionally NOT
                # used because high AI confidence can occur
                # on incorrect periodic aliases.
                #
                is_found = (
                    score >= GSPE_REJECTION_THRESHOLD
                )

                if is_found:

                    # ----------------------------------------
                    # ACCEPTED / PRESENT
                    # ----------------------------------------
                    found = 1

                else:

                    # ----------------------------------------
                    # REJECTED / ABSENT
                    # ----------------------------------------
                    found = 0

                    # For rejected cases, coordinates are not
                    # meaningful and must not be submitted as
                    # a localization result.
                    x = 0.0
                    y = 0.0

                    # Geometry is also not meaningful after
                    # rejection.
                    theta = 0.0
                    scale = 0.0

                # --------------------------------------------
                # APPEND OUTPUT ROW
                # --------------------------------------------
                rows.append({
                    "pair_id": pair_id,
                    "x": x,
                    "y": y,
                    "theta": theta,
                    "scale": scale,
                    "found": found,
                    "score": score,
                })

                # --------------------------------------------
                # PROGRESS OUTPUT
                # --------------------------------------------
                status = (
                    "FOUND"
                    if found
                    else "REJECT"
                )

                print(
                    f"[{status}] "
                    f"{pair_id} "
                    f"score={score:.6f} "
                    f"x={x:.3f} "
                    f"y={y:.3f}"
                )

            except Exception as exc:

                print(
                    f"[WARN] {pair_id}: {exc}"
                )

                # --------------------------------------------
                # PIPELINE FAILURE
                # --------------------------------------------
                rows.append({
                    "pair_id": pair_id,
                    "x": 0.0,
                    "y": 0.0,
                    "theta": 0.0,
                    "scale": 0.0,
                    "found": 0,
                    "score": 0.0,
                })

    # --------------------------------------------------------
    # WRITE FINAL PREDICTIONS CSV
    # --------------------------------------------------------
    with output_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=required_output_columns
        )

        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------
    found_count = sum(
        1
        for r in rows
        if int(r["found"]) == 1
    )

    rejected_count = len(rows) - found_count

    print()
    print("=" * 60)
    print("DRIFT-SENSE V2 REGISTRATION COMPLETE")
    print("=" * 60)
    print(
        f"Predictions written : {output_path}"
    )
    print(
        f"Pairs processed     : {len(rows)}"
    )
    print(
        f"Found               : {found_count}"
    )
    print(
        f"Rejected            : {rejected_count}"
    )
    print(
        f"GSPE threshold      : "
        f"{GSPE_REJECTION_THRESHOLD:.2f}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()