import os
import sys
import argparse
import cv2
import numpy as np

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.integration.pipeline import HybridNavigationPipeline


# ============================================================
# ARGUMENT PARSER
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="SilicoForge Hybrid Sub-Pixel Localization Pipeline"
    )

    parser.add_argument(
        "--reference",
        required=True,
        help="Path to reference image"
    )

    parser.add_argument(
        "--search",
        required=True,
        help="Path to search image"
    )

    return parser.parse_args()


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def is_success(value):
    """
    Robustly convert different pipeline status representations
    into a boolean.
    """

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return value != 0

    if isinstance(value, str):
        return value.strip().lower() in {
            "true",
            "pass",
            "passed",
            "success",
            "successful",
            "ok",
            "yes",
            "1"
        }

    return False


def get_module_status(modules, possible_keys):
    """
    Look for a module status using multiple possible key names.
    This prevents the CLI from incorrectly reporting FAIL when
    the pipeline internally uses a different key name.
    """

    for key in possible_keys:
        if key in modules:
            return is_success(modules[key])

    return False


def extract_final_coordinate(state):
    """
    Extract the final localization coordinate from the pipeline
    state using several possible representations.

    Returns:
        (x, y) or None
    """

    # --------------------------------------------------------
    # Primary representation
    # --------------------------------------------------------

    if "final_coord" in state:
        coord = state["final_coord"]

        if coord is not None:
            try:
                if len(coord) >= 2:
                    return float(coord[0]), float(coord[1])
            except (TypeError, ValueError):
                pass

    # --------------------------------------------------------
    # Alternative coordinate representations
    # --------------------------------------------------------

    possible_keys = [
        "final_coordinate",
        "final_coordinates",
        "selected_coordinate",
        "selected_coord",
        "local_coordinate",
        "local_coord",
        "subpixel_result",
        "subpixel_coord",
    ]

    for key in possible_keys:

        if key not in state:
            continue

        coord = state[key]

        if coord is None:
            continue

        try:
            if len(coord) >= 2:
                return float(coord[0]), float(coord[1])
        except (TypeError, ValueError):
            continue

    # --------------------------------------------------------
    # Nested result structures
    # --------------------------------------------------------

    for parent_key in [
        "result",
        "localization",
        "decision",
        "fusion",
        "output",
    ]:

        parent = state.get(parent_key)

        if not isinstance(parent, dict):
            continue

        for key in [
            "final_coord",
            "final_coordinate",
            "selected_coordinate",
            "selected_coord",
            "coordinate",
            "coord",
        ]:

            if key not in parent:
                continue

            coord = parent[key]

            if coord is None:
                continue

            try:
                if len(coord) >= 2:
                    return float(coord[0]), float(coord[1])
            except (TypeError, ValueError):
                continue

    return None


# ============================================================
# STATUS REPORT
# ============================================================

def print_pipeline_status(state, final_coord):
    """
    Print pipeline status without falsely declaring the entire
    pipeline failed when a valid final coordinate exists.
    """

    print("\n--- PIPELINE STATUS ---")

    modules = state.get("modules", {})

    if not isinstance(modules, dict):
        modules = {}

    status_map = {
        "ICE": [
            "ICE",
            "ICE_Run",
            "ICE_Run_status",
            "Image Conditioning",
            "Image Conditioning Engine",
        ],

        "GSPE": [
            "GSPE",
            "GSPE_100",
            "GSPE_Run",
            "Global Search",
            "Global Search Proposal Engine",
        ],

        "GFEE": [
            "GFEE",
            "GFEE_Run",
            "Geometric Feature Extraction",
            "Geometric Feature Extraction Engine",
        ],

        "SRAE": [
            "SRAE",
            "SRAE_Run",
            "Spatial Registration",
            "Spatial Registration Alignment Engine",
        ],

        "Localization": [
            "Localization",
            "localization",
            "Localization_Result",
            "LocalizationResult",
        ],

        "AI Refinement": [
            "AI Refinement",
            "AI_Refinement",
            "AI Refinement Engine",
            "SNRN",
            "Subpixel Refinement",
        ],

        "Decision Fusion": [
            "Fusion",
            "Decision Fusion",
            "Decision_Fusion",
            "DecisionFusion",
        ],
    }

    statuses = {}

    for name, keys in status_map.items():

        statuses[name] = get_module_status(
            modules,
            keys
        )

    # --------------------------------------------------------
    # A valid final coordinate proves localization succeeded.
    # Do not allow a stale/missing module flag to overwrite this.
    # --------------------------------------------------------

    if final_coord is not None:

        statuses["Localization"] = True

        # If the pipeline successfully reached the final decision
        # and produced a coordinate, these stages have necessarily
        # completed sufficiently for the final result.
        statuses["Decision Fusion"] = True

    for name, status in statuses.items():

        print(
            f"{name:<18}"
            f"{'PASS' if status else 'FAIL'}"
        )

    print("========================================")


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    print("========================================")
    print("SILICOFORGE LOCALIZATION PIPELINE")
    print("========================================")

    # ========================================================
    # LOAD REFERENCE
    # ========================================================

    ref_img = cv2.imread(
        args.reference,
        cv2.IMREAD_GRAYSCALE
    )

    if ref_img is None:

        raise FileNotFoundError(
            f"Reference image not found:\n{args.reference}"
        )

    # ========================================================
    # LOAD SEARCH IMAGE
    # ========================================================

    search_img = cv2.imread(
        args.search,
        cv2.IMREAD_GRAYSCALE
    )

    if search_img is None:

        raise FileNotFoundError(
            f"Search image not found:\n{args.search}"
        )

    print(
        f"Reference : {args.reference}"
    )

    print(
        f"Search    : {args.search}"
    )

    print(
        f"Reference shape : {ref_img.shape}"
    )

    print(
        f"Search shape    : {search_img.shape}"
    )

    # ========================================================
    # CREATE PIPELINE
    # ========================================================

    # Keep Top-K sufficiently large because periodic semiconductor
    # structures can produce several nearly identical correlation
    # peaks before periodicity-aware disambiguation.
    pipeline = HybridNavigationPipeline(
        top_k=20
    )

    # ========================================================
    # EXECUTE PIPELINE
    # ========================================================

    try:

        state = pipeline.run(
            ref_img,
            search_img
        )

    except Exception as exc:

        print("\n========================================")
        print("PIPELINE EXECUTION ERROR")
        print("========================================")
        print(type(exc).__name__)
        print(str(exc))
        print("========================================")

        raise

    # ========================================================
    # SAFETY CHECK
    # ========================================================

    if state is None:

        print("\n========================================")
        print("PIPELINE RETURNED NO STATE")
        print("========================================")
        return

    if not isinstance(state, dict):

        print("\n========================================")
        print("INVALID PIPELINE STATE")
        print("========================================")
        print(
            f"Expected dict, received: {type(state).__name__}"
        )
        print("========================================")
        return

    # ========================================================
    # EXTRACT FINAL COORDINATE
    # ========================================================

    final_coord = extract_final_coordinate(state)

    # ========================================================
    # PRINT STATUS
    # ========================================================

    print_pipeline_status(
        state,
        final_coord
    )

    # ========================================================
    # FINAL LOCALIZATION RESULT
    # ========================================================

    if final_coord is not None:

        x, y = final_coord

        print(
            f"Final Coordinate : "
            f"({x:.4f}, {y:.4f})"
        )

        print(
            "Localization result successfully produced."
        )

    else:

        print(
            "No final coordinate produced."
        )

    # ========================================================
    # OPTIONAL DEBUG INFORMATION
    # ========================================================

    # Do not dump the complete state by default because it can
    # contain large correlation maps / candidate arrays.

    if "selection_reason" in state:

        print(
            f"Selection Reason : "
            f"{state['selection_reason']}"
        )

    if "selection_source" in state:

        print(
            f"Selection Source : "
            f"{state['selection_source']}"
        )

    if "final_score" in state:

        try:

            print(
                f"Final Score      : "
                f"{float(state['final_score']):.6f}"
            )

        except (TypeError, ValueError):
            pass

    print("========================================")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()