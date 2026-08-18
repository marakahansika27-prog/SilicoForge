import os
import sys
import time
import logging
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np


# ---------------------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


logger = logging.getLogger("GSPE_Run")


# ---------------------------------------------------------------------
# NUMERICAL HELPERS
# ---------------------------------------------------------------------

def _safe_float(value, default=0.0):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return float(default)


def _clip(value, low, high):
    return max(low, min(value, high))


# ---------------------------------------------------------------------
# GSPE
# ---------------------------------------------------------------------

class GlobalSearchProposalEngine:
    """
    Global Search Proposal Engine.

    Purpose
    -------
    Generate a spatially diverse set of candidate locations using
    normalized cross correlation.

    Design
    ------
    1. Multi-scale template generation.
    2. Multi-rotation template generation.
    3. Raw NCC response.
    4. Low-frequency NCC response.
    5. Edge NCC response.
    6. Candidate extraction from every hypothesis.
    7. Duplicate suppression.
    8. Spatial diversity preservation.
    9. Large candidate pool returned to the downstream pipeline.

    Important
    ---------
    GSPE does NOT attempt to decide which periodic replica is the
    physical target. Its job is candidate recall.

    Therefore:
        candidate recall > aggressive Top-K suppression.

    The final decision belongs to downstream ranking/fusion.
    """

    def __init__(
        self,
        top_k: int = 20,
        nms_radius: int = 50,
        scale_hypotheses: List[float] = None,
        rotation_hypotheses: List[float] = None,
        candidate_multiplier: int = 4,
    ):

        self.top_k = max(1, int(top_k))
        self.nms_radius = max(1, int(nms_radius))

        self.scale_hypotheses = (
            list(scale_hypotheses)
            if scale_hypotheses is not None
            else [10.0]
        )

        self.rotation_hypotheses = (
            list(rotation_hypotheses)
            if rotation_hypotheses is not None
            else [0.0]
        )

        self.candidate_multiplier = max(
            1,
            int(candidate_multiplier)
        )

        # Do not let GSPE throw away candidates too early.
        self.return_limit = max(
            self.top_k * self.candidate_multiplier,
            80
        )

        self.stats = {}

    # -----------------------------------------------------------------
    # IMAGE PREPARATION
    # -----------------------------------------------------------------

    @staticmethod
    def _to_gray_uint8(image):

        if image is None:
            raise ValueError(
                "GSPE received None image."
            )

        arr = np.asarray(image)

        if arr.size == 0:
            raise ValueError(
                "GSPE received empty image."
            )

        if arr.ndim == 3:

            if arr.shape[2] == 1:
                arr = arr[:, :, 0]

            else:
                arr = cv2.cvtColor(
                    arr.astype(np.uint8),
                    cv2.COLOR_BGR2GRAY
                )

        arr = np.nan_to_num(
            arr,
            nan=0.0,
            posinf=255.0,
            neginf=0.0
        )

        if arr.dtype != np.uint8:

            mn = float(arr.min())
            mx = float(arr.max())

            if mx <= 1.0 and mn >= 0.0:
                arr = arr * 255.0

            elif mx > mn:
                arr = (
                    (arr - mn)
                    / (mx - mn)
                    * 255.0
                )

            else:
                arr = np.zeros_like(arr)

            arr = np.clip(
                arr,
                0,
                255
            ).astype(np.uint8)

        return arr

    # -----------------------------------------------------------------
    # TEMPLATE ROTATION
    # -----------------------------------------------------------------

    @staticmethod
    def _rotate_template(
        template,
        angle
    ):

        if abs(float(angle)) < 1e-9:
            return template

        h, w = template.shape[:2]

        center = (
            w / 2.0,
            h / 2.0
        )

        matrix = cv2.getRotationMatrix2D(
            center,
            float(angle),
            1.0
        )

        rotated = cv2.warpAffine(
            template,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        return rotated

    # -----------------------------------------------------------------
    # LOW FREQUENCY IMAGE
    # -----------------------------------------------------------------

    @staticmethod
    def _low_frequency(
        image
    ):

        image_f = image.astype(
            np.float32
        )

        # Preserve broad structure while removing
        # high-frequency noise.
        return cv2.GaussianBlur(
            image_f,
            (0, 0),
            sigmaX=2.0,
            sigmaY=2.0
        )

    # -----------------------------------------------------------------
    # EDGE IMAGE
    # -----------------------------------------------------------------

    @staticmethod
    def _edge_image(
        image
    ):

        gx = cv2.Sobel(
            image,
            cv2.CV_32F,
            1,
            0,
            ksize=3
        )

        gy = cv2.Sobel(
            image,
            cv2.CV_32F,
            0,
            1,
            ksize=3
        )

        magnitude = cv2.magnitude(
            gx,
            gy
        )

        max_value = float(
            magnitude.max()
        )

        if max_value > 1e-8:
            magnitude /= max_value

        return magnitude.astype(
            np.float32
        )

    # -----------------------------------------------------------------
    # NCC
    # -----------------------------------------------------------------

    @staticmethod
    def _match_template(
        search,
        template
    ):

        sh, sw = search.shape[:2]
        th, tw = template.shape[:2]

        if (
            th <= 1
            or tw <= 1
            or th > sh
            or tw > sw
        ):
            return None

        search_f = search.astype(
            np.float32
        )

        template_f = template.astype(
            np.float32
        )

        try:

            response = cv2.matchTemplate(
                search_f,
                template_f,
                cv2.TM_CCOEFF_NORMED
            )

        except cv2.error:

            return None

        response = np.nan_to_num(
            response,
            nan=-1.0,
            posinf=-1.0,
            neginf=-1.0
        )

        return response

    # -----------------------------------------------------------------
    # SUBPIXEL PEAK
    # -----------------------------------------------------------------

    @staticmethod
    def _subpixel_peak(
        response,
        x,
        y
    ):

        h, w = response.shape

        x = int(
            _clip(
                x,
                0,
                w - 1
            )
        )

        y = int(
            _clip(
                y,
                0,
                h - 1
            )
        )

        if (
            x <= 0
            or x >= w - 1
            or y <= 0
            or y >= h - 1
        ):
            return (
                float(x),
                float(y)
            )

        center = float(
            response[y, x]
        )

        left = float(
            response[y, x - 1]
        )

        right = float(
            response[y, x + 1]
        )

        up = float(
            response[y - 1, x]
        )

        down = float(
            response[y + 1, x]
        )

        denom_x = (
            left
            - 2.0 * center
            + right
        )

        denom_y = (
            up
            - 2.0 * center
            + down
        )

        if abs(denom_x) > 1e-8:

            dx = (
                0.5
                * (left - right)
                / denom_x
            )

        else:
            dx = 0.0

        if abs(denom_y) > 1e-8:

            dy = (
                0.5
                * (up - down)
                / denom_y
            )

        else:
            dy = 0.0

        dx = _clip(
            dx,
            -0.5,
            0.5
        )

        dy = _clip(
            dy,
            -0.5,
            0.5
        )

        return (
            float(x) + float(dx),
            float(y) + float(dy)
        )

    # -----------------------------------------------------------------
    # LOCAL PEAK EXTRACTION
    # -----------------------------------------------------------------

    @staticmethod
    def _extract_peaks(
        response,
        count,
        radius
    ):

        if response is None:
            return []

        work = response.copy()

        h, w = work.shape

        peaks = []

        radius = max(
            1,
            int(radius)
        )

        for _ in range(
            max(1, int(count))
        ):

            _, max_value, _, max_location = (
                cv2.minMaxLoc(work)
            )

            if not np.isfinite(
                max_value
            ):
                break

            if max_value <= -0.999:
                break

            px, py = max_location

            peaks.append(
                (
                    int(px),
                    int(py),
                    float(max_value)
                )
            )

            x1 = max(
                0,
                px - radius
            )

            x2 = min(
                w,
                px + radius + 1
            )

            y1 = max(
                0,
                py - radius
            )

            y2 = min(
                h,
                py + radius + 1
            )

            work[
                y1:y2,
                x1:x2
            ] = -1.0

        return peaks

    # -----------------------------------------------------------------
    # CANDIDATE KEY
    # -----------------------------------------------------------------

    @staticmethod
    def _candidate_distance(
        a,
        b
    ):

        ax = float(a["center_x"])
        ay = float(a["center_y"])

        bx = float(b["center_x"])
        by = float(b["center_y"])

        return float(
            np.hypot(
                ax - bx,
                ay - by
            )
        )

    # -----------------------------------------------------------------
    # DUPLICATE REMOVAL
    # -----------------------------------------------------------------

    def _deduplicate(
        self,
        candidates
    ):

        if not candidates:
            return []

        # Sort by strongest raw evidence.
        ordered = sorted(
            candidates,
            key=lambda c: (
                c["raw_score"],
                c["low_score"],
                c["edge_score"]
            ),
            reverse=True
        )

        unique = []

        # Small duplicate radius.
        # This is deliberately smaller than the spatial
        # diversity radius.
        duplicate_radius = max(
            3.0,
            min(
                12.0,
                self.nms_radius * 0.20
            )
        )

        for candidate in ordered:

            duplicate = False

            for existing in unique:

                if (
                    self._candidate_distance(
                        candidate,
                        existing
                    )
                    <= duplicate_radius
                ):
                    duplicate = True
                    break

            if not duplicate:
                unique.append(
                    candidate
                )

        return unique

    # -----------------------------------------------------------------
    # SPATIAL DIVERSITY
    # -----------------------------------------------------------------

    def _spatial_diversity(
        self,
        candidates
    ):

        if not candidates:
            return []

        ordered = sorted(
            candidates,
            key=lambda c: (
                c["raw_score"],
                c["low_score"],
                c["edge_score"]
            ),
            reverse=True
        )

        selected = []

        # Important:
        # Do not use the user supplied nms_radius directly here.
        # A 50 px NMS radius can unnecessarily destroy valid
        # neighboring periodic hypotheses.
        diversity_radius = max(
            20.0,
            min(
                40.0,
                self.nms_radius
            )
        )

        for candidate in ordered:

            keep = True

            for existing in selected:

                if (
                    self._candidate_distance(
                        candidate,
                        existing
                    )
                    < diversity_radius
                ):

                    keep = False
                    break

            if keep:

                selected.append(
                    candidate
                )

                if len(selected) >= self.return_limit:
                    break

        return selected

    # -----------------------------------------------------------------
    # MAIN
    # -----------------------------------------------------------------

    def run(
        self,
        inputs: Dict[str, Any]
    ):

        start_time = time.perf_counter()

        reference = inputs.get(
            "reference"
        )

        search = inputs.get(
            "search"
        )

        reference = self._to_gray_uint8(
            reference
        )

        search = self._to_gray_uint8(
            search
        )

        search_h, search_w = (
            search.shape[:2]
        )

        ref_h, ref_w = (
            reference.shape[:2]
        )

        # -------------------------------------------------------------
        # INPUT VALIDATION
        # -------------------------------------------------------------

        if (
            ref_h < 4
            or ref_w < 4
        ):
            raise ValueError(
                "Reference image is too small."
            )

        if (
            search_h < 4
            or search_w < 4
        ):
            raise ValueError(
                "Search image is too small."
            )

        # -------------------------------------------------------------
        # RESPONSE SURFACE ACCUMULATORS
        # -------------------------------------------------------------

        all_candidates = []

        best_raw_score = -1.0
        best_raw_location = (
            0,
            0
        )

        best_low_score = -1.0
        best_low_location = (
            0,
            0
        )

        best_hybrid_score = -1.0
        best_hybrid_location = (
            0,
            0
        )

        hypothesis_count = 0

        # -------------------------------------------------------------
        # SEARCH EVERY SCALE / ROTATION HYPOTHESIS
        # -------------------------------------------------------------

        print()
        print(
            "--- MULTI-HYPOTHESIS GSPE ---"
        )

        print(
            "Scale hypotheses :",
            self.scale_hypotheses
        )

        print(
            "Rotation hypotheses :",
            self.rotation_hypotheses
        )

        for scale in self.scale_hypotheses:

            scale = float(scale)

            if scale <= 0:
                continue

            new_w = max(
                4,
                int(
                    round(
                        ref_w / scale
                    )
                )
            )

            new_h = max(
                4,
                int(
                    round(
                        ref_h / scale
                    )
                )
            )

            if (
                new_w > search_w
                or new_h > search_h
            ):
                continue

            resized_reference = cv2.resize(
                reference,
                (
                    new_w,
                    new_h
                ),
                interpolation=cv2.INTER_AREA
            )

            for rotation in (
                self.rotation_hypotheses
            ):

                rotation = float(
                    rotation
                )

                hypothesis_count += 1

                template = (
                    self._rotate_template(
                        resized_reference,
                        rotation
                    )
                )

                # -------------------------------------------------
                # RAW
                # -------------------------------------------------

                raw_response = (
                    self._match_template(
                        search,
                        template
                    )
                )

                if raw_response is None:
                    continue

                # -------------------------------------------------
                # LOW FREQUENCY
                # -------------------------------------------------

                low_search = (
                    self._low_frequency(
                        search
                    )
                )

                low_template = (
                    self._low_frequency(
                        template
                    )
                )

                low_response = (
                    self._match_template(
                        low_search,
                        low_template
                    )
                )

                if low_response is None:
                    continue

                # -------------------------------------------------
                # EDGE
                # -------------------------------------------------

                edge_search = (
                    self._edge_image(
                        search
                    )
                )

                edge_template = (
                    self._edge_image(
                        template
                    )
                )

                edge_response = (
                    self._match_template(
                        edge_search,
                        edge_template
                    )
                )

                if edge_response is None:
                    continue

                # -------------------------------------------------
                # GLOBAL DIAGNOSTICS
                # -------------------------------------------------

                _, raw_max, _, raw_loc = (
                    cv2.minMaxLoc(
                        raw_response
                    )
                )

                _, low_max, _, low_loc = (
                    cv2.minMaxLoc(
                        low_response
                    )
                )

                _, edge_max, _, edge_loc = (
                    cv2.minMaxLoc(
                        edge_response
                    )
                )

                if raw_max > best_raw_score:

                    best_raw_score = (
                        float(raw_max)
                    )

                    best_raw_location = (
                        int(raw_loc[0]),
                        int(raw_loc[1])
                    )

                if low_max > best_low_score:

                    best_low_score = (
                        float(low_max)
                    )

                    best_low_location = (
                        int(low_loc[0]),
                        int(low_loc[1])
                    )

                # -------------------------------------------------
                # HYBRID SURFACE
                #
                # RAW remains dominant.
                # Low-frequency and edge surfaces are diagnostics
                # and secondary evidence.
                # -------------------------------------------------

                hybrid_response = (
                    0.65 * raw_response
                    +
                    0.25 * low_response
                    +
                    0.10 * edge_response
                )

                _, hybrid_max, _, hybrid_loc = (
                    cv2.minMaxLoc(
                        hybrid_response
                    )
                )

                if (
                    hybrid_max
                    > best_hybrid_score
                ):

                    best_hybrid_score = (
                        float(hybrid_max)
                    )

                    best_hybrid_location = (
                        int(hybrid_loc[0]),
                        int(hybrid_loc[1])
                    )

                # -------------------------------------------------
                # PEAK EXTRACTION
                #
                # Extract more peaks than final output requires.
                # This protects periodic GT candidates.
                # -------------------------------------------------

                peak_count = max(
                    12,
                    self.top_k
                )

                raw_peaks = (
                    self._extract_peaks(
                        raw_response,
                        peak_count,
                        max(
                            8,
                            int(
                                self.nms_radius * 0.50
                            )
                        )
                    )
                )

                low_peaks = (
                    self._extract_peaks(
                        low_response,
                        peak_count,
                        max(
                            8,
                            int(
                                self.nms_radius * 0.50
                            )
                        )
                    )
                )

                edge_peaks = (
                    self._extract_peaks(
                        edge_response,
                        peak_count,
                        max(
                            8,
                            int(
                                self.nms_radius * 0.50
                            )
                        )
                    )
                )

                # -------------------------------------------------
                # ADD RAW PEAKS
                # -------------------------------------------------

                for px, py, raw_score in raw_peaks:

                    sub_x, sub_y = (
                        self._subpixel_peak(
                            raw_response,
                            px,
                            py
                        )
                    )

                    tl_x = (
                        sub_x
                    )

                    tl_y = (
                        sub_y
                    )

                    center_x = (
                        tl_x
                        + new_w / 2.0
                    )

                    center_y = (
                        tl_y
                        + new_h / 2.0
                    )

                    ix = int(
                        _clip(
                            round(px),
                            0,
                            raw_response.shape[1] - 1
                        )
                    )

                    iy = int(
                        _clip(
                            round(py),
                            0,
                            raw_response.shape[0] - 1
                        )
                    )

                    low_score = (
                        float(
                            low_response[iy, ix]
                        )
                    )

                    edge_score = (
                        float(
                            edge_response[iy, ix]
                        )
                    )

                    all_candidates.append(
                        {
                            "center_x":
                                float(center_x),

                            "center_y":
                                float(center_y),

                            "raw_x":
                                float(sub_x),

                            "raw_y":
                                float(sub_y),

                            "raw_score":
                                float(raw_score),

                            "low_score":
                                low_score,

                            "edge_score":
                                edge_score,

                            "scale":
                                float(scale),

                            "rotation":
                                float(rotation),

                            "width":
                                int(new_w),

                            "height":
                                int(new_h),

                            "source":
                                "RAW_NCC",
                        }
                    )

                # -------------------------------------------------
                # ADD LOW-FREQUENCY PEAKS
                # -------------------------------------------------

                for px, py, low_score in low_peaks:

                    sub_x, sub_y = (
                        self._subpixel_peak(
                            low_response,
                            px,
                            py
                        )
                    )

                    center_x = (
                        sub_x
                        + new_w / 2.0
                    )

                    center_y = (
                        sub_y
                        + new_h / 2.0
                    )

                    ix = int(
                        _clip(
                            round(px),
                            0,
                            raw_response.shape[1] - 1
                        )
                    )

                    iy = int(
                        _clip(
                            round(py),
                            0,
                            raw_response.shape[0] - 1
                        )
                    )

                    raw_score = (
                        float(
                            raw_response[iy, ix]
                        )
                    )

                    edge_score = (
                        float(
                            edge_response[iy, ix]
                        )
                    )

                    all_candidates.append(
                        {
                            "center_x":
                                float(center_x),

                            "center_y":
                                float(center_y),

                            "raw_x":
                                float(sub_x),

                            "raw_y":
                                float(sub_y),

                            "raw_score":
                                raw_score,

                            "low_score":
                                float(low_score),

                            "edge_score":
                                edge_score,

                            "scale":
                                float(scale),

                            "rotation":
                                float(rotation),

                            "width":
                                int(new_w),

                            "height":
                                int(new_h),

                            "source":
                                "LOW_FREQ_NCC",
                        }
                    )

                # -------------------------------------------------
                # ADD EDGE PEAKS
                # -------------------------------------------------

                for px, py, edge_score in edge_peaks:

                    sub_x, sub_y = (
                        self._subpixel_peak(
                            edge_response,
                            px,
                            py
                        )
                    )

                    center_x = (
                        sub_x
                        + new_w / 2.0
                    )

                    center_y = (
                        sub_y
                        + new_h / 2.0
                    )

                    ix = int(
                        _clip(
                            round(px),
                            0,
                            raw_response.shape[1] - 1
                        )
                    )

                    iy = int(
                        _clip(
                            round(py),
                            0,
                            raw_response.shape[0] - 1
                        )
                    )

                    raw_score = (
                        float(
                            raw_response[iy, ix]
                        )
                    )

                    low_score = (
                        float(
                            low_response[iy, ix]
                        )
                    )

                    all_candidates.append(
                        {
                            "center_x":
                                float(center_x),

                            "center_y":
                                float(center_y),

                            "raw_x":
                                float(sub_x),

                            "raw_y":
                                float(sub_y),

                            "raw_score":
                                raw_score,

                            "low_score":
                                low_score,

                            "edge_score":
                                float(edge_score),

                            "scale":
                                float(scale),

                            "rotation":
                                float(rotation),

                            "width":
                                int(new_w),

                            "height":
                                int(new_h),

                            "source":
                                "EDGE_NCC",
                        }
                    )

        # -------------------------------------------------------------
        # DEDUPLICATION
        # -------------------------------------------------------------

        deduplicated = (
            self._deduplicate(
                all_candidates
            )
        )

        # -------------------------------------------------------------
        # SPATIAL DIVERSITY
        # -------------------------------------------------------------

        diverse_candidates = (
            self._spatial_diversity(
                deduplicated
            )
        )

        # -------------------------------------------------------------
        # FINAL ORDER
        #
        # Raw NCC remains the primary ordering.
        # Low/edge only break extremely close ties.
        # -------------------------------------------------------------

        diverse_candidates.sort(
            key=lambda c: (
                c["raw_score"],
                c["low_score"],
                c["edge_score"]
            ),
            reverse=True
        )

        final_candidates = (
            diverse_candidates[
                :self.return_limit
            ]
        )

        # -------------------------------------------------------------
        # CONVERT TO COMPATIBLE BOX FORMAT
        # -------------------------------------------------------------

        boxes = []
        scores = []

        for candidate in final_candidates:

            cx = float(
                candidate["center_x"]
            )

            cy = float(
                candidate["center_y"]
            )

            w = int(
                candidate["width"]
            )

            h = int(
                candidate["height"]
            )

            # Convert center to top-left.
            x = (
                cx - w / 2.0
            )

            y = (
                cy - h / 2.0
            )

            x = _clip(
                x,
                0.0,
                float(search_w - w)
                if search_w >= w
                else 0.0
            )

            y = _clip(
                y,
                0.0,
                float(search_h - h)
                if search_h >= h
                else 0.0
            )

            boxes.append(
                (
                    float(x),
                    float(y),
                    int(w),
                    int(h),
                    float(
                        candidate["scale"]
                    ),
                    float(
                        candidate["rotation"]
                    )
                )
            )

            scores.append(
                float(
                    candidate["raw_score"]
                )
            )

        # -------------------------------------------------------------
        # SORT AGAIN BY RAW NCC
        # -------------------------------------------------------------

        combined = sorted(
            zip(
                boxes,
                scores,
                final_candidates
            ),
            key=lambda item: (
                item[1],
                item[2]["low_score"],
                item[2]["edge_score"]
            ),
            reverse=True
        )

        boxes = [
            item[0]
            for item in combined
        ]

        scores = [
            float(item[1])
            for item in combined
        ]

        final_candidates = [
            item[2]
            for item in combined
        ]

        # -------------------------------------------------------------
        # DIAGNOSTICS
        # -------------------------------------------------------------

        print()
        print(
            "--- GSPE MULTI-HYPOTHESIS DIAGNOSTIC ---"
        )

        print(
            f"Hypotheses Evaluated : "
            f"{hypothesis_count}"
        )

        print(
            f"Raw Candidate Pool   : "
            f"{len(all_candidates)}"
        )

        print()
        print(
            "--- GSPE CANDIDATE DIVERSITY ---"
        )

        print(
            f"Independent candidates : "
            f"{len(all_candidates)}"
        )

        print(
            f"After duplicate removal: "
            f"{len(deduplicated)}"
        )

        print(
            f"After spatial diversity: "
            f"{len(diverse_candidates)}"
        )

        print(
            f"Final candidates       : "
            f"{len(boxes)}"
        )

        if scores:

            print()
            print(
                "--- RAW NCC GLOBAL WINNER ---"
            )

            best_box = boxes[0]

            best_cx = (
                best_box[0]
                + best_box[2] / 2.0
            )

            best_cy = (
                best_box[1]
                + best_box[3] / 2.0
            )

            print(
                f"Center     : "
                f"({best_cx:.4f}, "
                f"{best_cy:.4f})"
            )

            print(
                f"Score      : "
                f"{scores[0]:.6f}"
            )

            print(
                f"Scale      : "
                f"{best_box[4]:.4f}"
            )

            print(
                f"Rotation   : "
                f"{best_box[5]:.4f}"
            )

        print()
        print(
            "--- TOP DIVERSE GSPE CANDIDATES ---"
        )

        for rank, (
            box,
            score,
            candidate
        ) in enumerate(
            zip(
                boxes[:20],
                scores[:20],
                final_candidates[:20]
            ),
            start=1
        ):

            cx = (
                box[0]
                + box[2] / 2.0
            )

            cy = (
                box[1]
                + box[3] / 2.0
            )

            print(
                f"Rank {rank:02d}: "
                f"center=({cx:.3f}, "
                f"{cy:.3f}) "
                f"RAW={score:.6f} "
                f"LOW={candidate['low_score']:.6f} "
                f"EDGE={candidate['edge_score']:.6f} "
                f"scale={box[4]:.3f} "
                f"rot={box[5]:.3f} "
                f"source={candidate['source']}"
            )

        # -------------------------------------------------------------
        # RESPONSE MAP DIAGNOSTIC
        # -------------------------------------------------------------

        print()
        print(
            "--- GSPE RESPONSE MAP DIAGNOSTIC ---"
        )

        print(
            f"Best Raw NCC        : "
            f"{best_raw_score:.6f}"
        )

        print(
            f"Raw Map Peak        : "
            f"{best_raw_location}"
        )

        print(
            f"Best LowFreq NCC    : "
            f"{best_low_score:.6f}"
        )

        print(
            f"LowFreq Map Peak    : "
            f"{best_low_location}"
        )

        print(
            f"Best Hybrid NCC     : "
            f"{best_hybrid_score:.6f}"
        )

        print(
            f"Hybrid Map Peak     : "
            f"{best_hybrid_location}"
        )

        # -------------------------------------------------------------
        # RUNTIME
        # -------------------------------------------------------------

        runtime_ms = (
            time.perf_counter()
            - start_time
        ) * 1000.0

        self.stats = {

            "runtime_ms":
                float(runtime_ms),

            "hypotheses_evaluated":
                int(hypothesis_count),

            "raw_candidate_pool":
                int(len(all_candidates)),

            "deduplicated_candidates":
                int(len(deduplicated)),

            "diverse_candidates":
                int(len(diverse_candidates)),

            "final_candidates":
                int(len(boxes)),

            "raw_top1_score":
                float(
                    best_raw_score
                ),

            "lowfreq_top1_score":
                float(
                    best_low_score
                ),

            "hybrid_top1_score":
                float(
                    best_hybrid_score
                ),

            "raw_top1_location":
                tuple(
                    best_raw_location
                ),

            "lowfreq_top1_location":
                tuple(
                    best_low_location
                ),

            "hybrid_top1_location":
                tuple(
                    best_hybrid_location
                ),

            "raw_boxes":
                boxes.copy(),

            "raw_scores":
                scores.copy(),
        }

        logger.info(
            "[GSPE_Run] Executed in %.2f ms",
            runtime_ms
        )

        # -------------------------------------------------------------
        # RETURN
        # -------------------------------------------------------------

        return {

            "boxes":
                boxes,

            "scores":
                scores,

            "heatmap":
                None,

            "stats":
                self.stats,

            "res_raw":
                None,

            "res_lowfreq":
                None,

            "res_hybrid":
                None,

            "candidates":
                final_candidates,
        }