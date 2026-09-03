from pathlib import Path
import csv
import cv2


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "demo_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def read_prediction(csv_path):
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(f"No prediction found in {csv_path}")

    return rows[0]


def draw_result(
    pair_csv,
    prediction_csv,
    output_name,
    title,
):
    pair = read_prediction(pair_csv)
    pred = read_prediction(prediction_csv)

    reference_path = ROOT / pair["reference"]
    search_path = ROOT / pair["search"]

    reference = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
    search = cv2.imread(str(search_path), cv2.IMREAD_GRAYSCALE)

    if reference is None:
        raise FileNotFoundError(f"Reference image not found: {reference_path}")

    if search is None:
        raise FileNotFoundError(f"Search image not found: {search_path}")

    found = int(float(pred["found"]))
    score = float(pred["score"])
    x = float(pred["x"])
    y = float(pred["y"])

    # Convert grayscale images to displayable BGR images.
    ref_display = cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR)
    search_display = cv2.cvtColor(search, cv2.COLOR_GRAY2BGR)

    # ------------------------------------------------------------
    # Draw result on search image
    # ------------------------------------------------------------
    if found == 1:
        # Detection center
        center = (int(round(x)), int(round(y)))

        # Reference dimensions give us a useful visualization box.
        h, w = reference.shape[:2]
        half_w = max(20, w // 2)
        half_h = max(20, h // 2)

        x1 = max(0, center[0] - half_w)
        y1 = max(0, center[1] - half_h)
        x2 = min(search.shape[1] - 1, center[0] + half_w)
        y2 = min(search.shape[0] - 1, center[1] + half_h)

        cv2.rectangle(
            search_display,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            4,
        )

        # Crosshair
        cv2.drawMarker(
            search_display,
            center,
            (0, 255, 0),
            cv2.MARKER_CROSS,
            50,
            4,
        )

        decision = "FOUND"
        decision_color = (0, 255, 0)

    else:
        decision = "NOT FOUND"
        decision_color = (0, 0, 255)

    # ------------------------------------------------------------
    # Resize both images for a clean presentation
    # ------------------------------------------------------------
    panel_width = 800

    def resize_to_width(img, width):
        scale = width / img.shape[1]
        height = int(img.shape[0] * scale)
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    ref_display = resize_to_width(ref_display, panel_width)
    search_display = resize_to_width(search_display, panel_width)

    # ------------------------------------------------------------
    # Add labels
    # ------------------------------------------------------------
    cv2.rectangle(
        ref_display,
        (0, 0),
        (panel_width, 65),
        (15, 30, 55),
        -1,
    )

    cv2.putText(
        ref_display,
        "REFERENCE IMAGE",
        (25, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.rectangle(
        search_display,
        (0, 0),
        (panel_width, 65),
        (15, 30, 55),
        -1,
    )

    cv2.putText(
        search_display,
        "SEARCH IMAGE",
        (25, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # ------------------------------------------------------------
    # If the image was resized, scale the displayed coordinates
    # for the crosshair/box.
    #
    # We already drew the result before resizing, so the geometry
    # remains correctly scaled automatically.
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # Build information panel
    # ------------------------------------------------------------
    info_height = 190

    info = 255 * (
        cv2.UMat(info_height, panel_width, cv2.CV_8UC1).get()
    )
    info = cv2.cvtColor(info, cv2.COLOR_GRAY2BGR)

    # Dark background
    info[:] = (12, 24, 42)

    cv2.putText(
        info,
        "SilicoForge Phase 2",
        (25, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        info,
        f"Decision: {decision}",
        (25, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        decision_color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        info,
        f"Score: {score:.6f}",
        (25, 122),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if found == 1:
        cv2.putText(
            info,
            f"Predicted Center: ({x:.3f}, {y:.3f})",
            (25, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            info,
            "Pose output suppressed after rejection",
            (25, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # ------------------------------------------------------------
    # Combine vertically
    # ------------------------------------------------------------
    result = cv2.vconcat([
        ref_display,
        search_display,
        info,
    ])

    # Add title bar
    title_bar_height = 80

    title_bar = 255 * (
        cv2.UMat(title_bar_height, panel_width, cv2.CV_8UC1).get()
    )
    title_bar = cv2.cvtColor(title_bar, cv2.COLOR_GRAY2BGR)
    title_bar[:] = (5, 18, 35)

    cv2.putText(
        title_bar,
        title,
        (25, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    result = cv2.vconcat([
        title_bar,
        result,
    ])

    output_path = OUTPUT_DIR / output_name
    cv2.imwrite(str(output_path), result)

    print(f"Created: {output_path}")


def main():
    draw_result(
        ROOT / "pairs_demo.csv",
        ROOT / "predictions_demo.csv",
        "present_result.png",
        "PRESENT CASE — LOCALIZATION RESULT",
    )

    draw_result(
        ROOT / "pairs_absent_demo.csv",
        ROOT / "predictions_absent_demo.csv",
        "absent_result.png",
        "ABSENT CASE — REJECTION RESULT",
    )

    print()
    print("Demo visualizations created successfully.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
    