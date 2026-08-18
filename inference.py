import sys
from pathlib import Path

# Make project modules importable
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.integration.pipeline_backup_v2_ai import HybridNavigationPipeline


def main():
    if len(sys.argv) != 3:
        print("Usage: python inference.py <reference_image> <search_image>")
        sys.exit(1)

    reference_path = Path(sys.argv[1])
    search_path = Path(sys.argv[2])

    if not reference_path.exists():
        print(f"Error: reference image not found: {reference_path}")
        sys.exit(1)

    if not search_path.exists():
        print(f"Error: search image not found: {search_path}")
        sys.exit(1)

    import cv2
    import contextlib
    import io

    ref_img = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
    search_img = cv2.imread(str(search_path), cv2.IMREAD_GRAYSCALE)

    if ref_img is None:
        print(f"Error: Failed to load reference image (is it a valid image?)")
        sys.exit(1)
        
    if search_img is None:
        print(f"Error: Failed to load search image (is it a valid image?)")
        sys.exit(1)

    pipeline = HybridNavigationPipeline()
    
    # Suppress verbose prints unless requested (but for standard evaluator output, we just want the coord)
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        result = pipeline.run(ref_img, search_img)

    # The verified V2 pipeline exposes the final coordinate here.
    if 'error' in result:
        print(f"Pipeline error: {result['error']}")
        sys.exit(1)
        
    coord = result.get("final_coord")

    if coord is None:
        raise RuntimeError(
            "Localization pipeline did not return 'final_coord'. "
            f"Available keys: {list(result.keys())}"
        )

    x, y = coord

    # Evaluator-facing output.
    print(f"({float(x):.4f}, {float(y):.4f})")


if __name__ == "__main__":
    main()