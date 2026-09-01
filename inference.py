import sys
from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parent

# Make project modules importable
sys.path.insert(0, str(ROOT))

from src.integration.pipeline_backup_v2_ai import HybridNavigationPipeline


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Validate command-line arguments
    # --------------------------------------------------------

    if len(sys.argv) != 3:
        print(
            "Usage: python inference.py "
            "<reference_image> <search_image>"
        )
        sys.exit(1)

    reference_path = Path(sys.argv[1])
    search_path = Path(sys.argv[2])

    # --------------------------------------------------------
    # Validate input files
    # --------------------------------------------------------

    if not reference_path.exists():
        print(
            f"Error: reference image not found: "
            f"{reference_path}"
        )
        sys.exit(1)

    if not search_path.exists():
        print(
            f"Error: search image not found: "
            f"{search_path}"
        )
        sys.exit(1)

    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    import cv2

    ref_img = cv2.imread(
        str(reference_path),
        cv2.IMREAD_GRAYSCALE
    )

    search_img = cv2.imread(
        str(search_path),
        cv2.IMREAD_GRAYSCALE
    )

    if ref_img is None:
        print(
            "Error: Failed to load reference image "
            "(is it a valid image?)"
        )
        sys.exit(1)

    if search_img is None:
        print(
            "Error: Failed to load search image "
            "(is it a valid image?)"
        )
        sys.exit(1)

    # --------------------------------------------------------
    # Initialize production pipeline
    # --------------------------------------------------------

    pipeline = HybridNavigationPipeline()

    # --------------------------------------------------------
    # Run inference
    #
    # The production pipeline is verbose internally.
    # Suppress its stdout so that the evaluator receives
    # only the final coordinate.
    # --------------------------------------------------------

    import contextlib
    import io

    captured_stdout = io.StringIO()

    try:

        with contextlib.redirect_stdout(captured_stdout):

            result = pipeline.run(
                ref_img,
                search_img
            )

    except Exception as exc:

        print(
            f"Pipeline error: {exc}"
        )
        sys.exit(1)

    # --------------------------------------------------------
    # Validate pipeline result
    #
    # IMPORTANT:
    # The pipeline always contains the "error" key.
    # A normal successful result contains:
    #
    #     "error": None
    #
    # Therefore we must check the VALUE, not the presence
    # of the key.
    # --------------------------------------------------------

    if result is None:
        print(
            "Pipeline error: pipeline returned None"
        )
        sys.exit(1)

    pipeline_error = result.get("error")

    if pipeline_error is not None:
        print(
            f"Pipeline error: {pipeline_error}"
        )
        sys.exit(1)

    # --------------------------------------------------------
    # Extract final coordinate
    # --------------------------------------------------------

    coord = result.get("final_coord")

    if coord is None:
        print(
            "Pipeline error: Localization pipeline "
            "did not return 'final_coord'."
        )
        print(
            f"Available keys: {list(result.keys())}"
        )
        sys.exit(1)

    # --------------------------------------------------------
    # Validate coordinate structure
    # --------------------------------------------------------

    try:

        x, y = coord

        x = float(x)
        y = float(y)

    except (TypeError, ValueError):

        print(
            "Pipeline error: invalid final coordinate "
            f"returned by pipeline: {coord}"
        )
        sys.exit(1)

    # --------------------------------------------------------
    # Evaluator-facing output
    #
    # IMPORTANT:
    # Keep this as the ONLY normal stdout output.
    # --------------------------------------------------------

    print(
        f"({x:.4f}, {y:.4f})"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()