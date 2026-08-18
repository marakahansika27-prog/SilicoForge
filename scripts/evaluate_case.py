import os
import sys
import json
import argparse
import cv2
import numpy as np

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.integration.pipeline import HybridNavigationPipeline


def load_metadata(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def euclidean_error(pred, gt):
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)

    return float(np.linalg.norm(pred - gt))


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate one SilicoForge localization case."
    )

    parser.add_argument(
        "--reference",
        required=True
    )

    parser.add_argument(
        "--search",
        required=True
    )

    parser.add_argument(
        "--metadata",
        required=True
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # Load images
    # --------------------------------------------------

    reference = cv2.imread(
        args.reference,
        cv2.IMREAD_GRAYSCALE
    )

    search = cv2.imread(
        args.search,
        cv2.IMREAD_GRAYSCALE
    )

    if reference is None:
        raise FileNotFoundError(
            f"Reference not found: {args.reference}"
        )

    if search is None:
        raise FileNotFoundError(
            f"Search image not found: {args.search}"
        )

    # --------------------------------------------------
    # Load ground truth
    # --------------------------------------------------

    metadata = load_metadata(args.metadata)

    gt = np.array(
        [
            float(metadata["gt_x"]),
            float(metadata["gt_y"])
        ],
        dtype=np.float32
    )

    # --------------------------------------------------
    # Run pipeline
    # --------------------------------------------------

    pipeline = HybridNavigationPipeline()

    state = pipeline.run(
        reference,
        search
    )

    if "error" in state:
        print("\nPIPELINE FAILED")
        print(state["error"])
        sys.exit(1)

    # --------------------------------------------------
    # Coordinates
    # --------------------------------------------------

    final_coord = np.asarray(
        state["final_coord"],
        dtype=np.float32
    )

    macro_coord = np.asarray(
        state.get(
            "macro_coord",
            final_coord
        ),
        dtype=np.float32
    )

    local_coord = np.asarray(
        state.get(
            "local_coord",
            final_coord
        ),
        dtype=np.float32
    )

    subpixel_coord = np.asarray(
        state.get(
            "subpixel_coord",
            final_coord
        ),
        dtype=np.float32
    )

    # --------------------------------------------------
    # Errors
    # --------------------------------------------------

    macro_error = euclidean_error(
        macro_coord,
        gt
    )

    local_error = euclidean_error(
        local_coord,
        gt
    )

    subpixel_error = euclidean_error(
        subpixel_coord,
        gt
    )

    # --------------------------------------------------
    # Improvement
    # --------------------------------------------------

    if macro_error > 1e-8:

        improvement = (
            (macro_error - subpixel_error)
            / macro_error
        ) * 100.0

    else:

        improvement = 0.0

    # --------------------------------------------------
    # Output
    # --------------------------------------------------

    print("\n")
    print("=" * 60)
    print("SILICOFORGE BASELINE EVALUATION")
    print("=" * 60)

    print(
        f"Case ID          : "
        f"{metadata.get('case_id', 'UNKNOWN')}"
    )

    print(
        f"Architecture     : "
        f"{metadata.get('architecture', 'UNKNOWN')}"
    )

    print(
        f"Difficulty       : "
        f"{metadata.get('difficulty', 'UNKNOWN')}"
    )

    print(
        f"Spatial Region   : "
        f"{metadata.get('spatial_region', 'UNKNOWN')}"
    )

    print("-" * 60)

    print(
        f"Ground Truth     : "
        f"({gt[0]:.4f}, {gt[1]:.4f})"
    )

    print(
        f"GSPE Coordinate  : "
        f"({macro_coord[0]:.4f}, "
        f"{macro_coord[1]:.4f})"
    )

    print(
        f"Local Coordinate : "
        f"({local_coord[0]:.4f}, "
        f"{local_coord[1]:.4f})"
    )

    print(
        f"Subpixel Result  : "
        f"({subpixel_coord[0]:.4f}, "
        f"{subpixel_coord[1]:.4f})"
    )

    print("-" * 60)

    print(
        f"GSPE Error       : "
        f"{macro_error:.4f} px"
    )

    print(
        f"Local Error      : "
        f"{local_error:.4f} px"
    )

    print(
        f"Subpixel Error   : "
        f"{subpixel_error:.4f} px"
    )

    print(
        f"Refinement Gain  : "
        f"{improvement:.2f}%"
    )

    print(
        f"Runtime          : "
        f"{state.get('runtime', 0.0) * 1000:.2f} ms"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()