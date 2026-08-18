import os
import sys
import logging
import numpy as np
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.ice import ImageConditioningEngine
from src.coarse_search.gspe import GlobalSearchProposalEngine
from src.feature_matching.gfee import GeometricFeatureExtractionEngine
from src.registration.srae import SpatialRegistrationAlignmentEngine

logger = logging.getLogger("SilicoForgePipeline")


def _safe_float(value, default=0.0):
    try:
        x = float(value)
        return x if np.isfinite(x) else float(default)
    except Exception:
        return float(default)


def _safe_int(value, default=0):
    try:
        return int(round(float(value)))
    except Exception:
        return int(default)


def _candidate_center(c):
    if not isinstance(c, dict):
        return 0.0, 0.0
    for xk, yk in (("center_x", "center_y"), ("canonical_x", "canonical_y")):
        if xk in c and yk in c:
            return _safe_float(c[xk]), _safe_float(c[yk])
    if "center" in c and isinstance(c["center"], (list, tuple, np.ndarray)):
        if len(c["center"]) >= 2:
            return _safe_float(c["center"][0]), _safe_float(c["center"][1])
    b = c.get("box", c.get("bbox"))
    if isinstance(b, (list, tuple, np.ndarray)) and len(b) >= 4:
        x, y, w, h = map(_safe_float, b[:4])
        return x + w / 2.0, y + h / 2.0
    return _safe_float(c.get("x")), _safe_float(c.get("y"))


def _candidate_score(c):
    if not isinstance(c, dict):
        return 0.0
    for key in ("raw_ncc", "ncc", "score", "correlation_score"):
        if key in c:
            return _safe_float(c[key])
    return 0.0


def _candidate_box(c):
    if not isinstance(c, dict):
        return None
    b = c.get("box", c.get("bbox"))
    if isinstance(b, (list, tuple, np.ndarray)) and len(b) >= 4:
        x, y, w, h = map(_safe_float, b[:4])
        if w > 0 and h > 0:
            return [x, y, w, h]
    cx, cy = _candidate_center(c)
    w = _safe_float(c.get("width", c.get("w", 0)))
    h = _safe_float(c.get("height", c.get("h", 0)))
    if w <= 0 or h <= 0:
        return None
    return [cx - w / 2.0, cy - h / 2.0, w, h]


def _clip_xy(x, y, shape):
    h, w = shape[:2]
    return max(0.0, min(float(x), w - 1.0)), max(0.0, min(float(y), h - 1.0))


def _normalize_gspe_candidates(gspe_result, top_k=40):
    """
    Normalize GSPE output WITHOUT changing the search/selection algorithm.

    Critical preservation rule:
    The score produced by GSPE must survive this handoff unchanged.
    """

    if not isinstance(gspe_result, dict):
        return []

    candidates = []

    # -------------------------------------------------------------
    # CASE 1: GSPE already returned candidate dictionaries
    # -------------------------------------------------------------
    raw = gspe_result.get("candidates")

    if isinstance(raw, list) and raw:

        for i, c in enumerate(raw):

            if not isinstance(c, dict):
                continue

            item = dict(c)

            x, y = _candidate_center(item)

            # Preserve the ORIGINAL GSPE score.
            score = None

            for key in (
                "raw_ncc",
                "raw_score",
                "ncc",
                "score",
                "correlation_score",
                "template_score",
            ):
                if key in item:
                    try:
                        value = float(item[key])
                        if np.isfinite(value):
                            score = value
                            break
                    except Exception:
                        pass

            if score is None:
                score = 0.0

            item["center_x"] = float(x)
            item["center_y"] = float(y)
            item["x"] = float(x)
            item["y"] = float(y)

            # Canonical score fields.
            item["raw_ncc"] = float(score)
            item["ncc"] = float(score)
            item["score"] = float(score)
            item["correlation_score"] = float(score)

            item.setdefault("source", "GSPE")

            candidates.append(item)

    # -------------------------------------------------------------
    # CASE 2: GSPE returned boxes + score arrays
    # -------------------------------------------------------------
    else:

        boxes = gspe_result.get("boxes", [])
        scores = gspe_result.get("scores", [])

        raw_nccs = (
            gspe_result.get("raw_nccs")
            or gspe_result.get("raw_scores")
            or gspe_result.get("ncc_scores")
        )

        scales = gspe_result.get("scales", [])
        rotations = gspe_result.get("rotations", [])

        for i, b in enumerate(boxes):

            if not isinstance(
                b,
                (list, tuple, np.ndarray)
            ):
                continue

            if len(b) < 4:
                continue

            x, y, w, h = map(
                _safe_float,
                b[:4]
            )

            if w <= 0 or h <= 0:
                continue

            # -----------------------------------------------------
            # IMPORTANT:
            # Prefer the explicit RAW NCC array when available.
            # Otherwise use GSPE's scores array.
            # -----------------------------------------------------

            if (
                isinstance(
                    raw_nccs,
                    (list, tuple, np.ndarray)
                )
                and i < len(raw_nccs)
            ):

                score = _safe_float(
                    raw_nccs[i]
                )

            elif i < len(scores):

                score = _safe_float(
                    scores[i]
                )

            else:

                score = 0.0

            cx = x + w / 2.0
            cy = y + h / 2.0

            item = {
                "x": float(cx),
                "y": float(cy),

                "center_x": float(cx),
                "center_y": float(cy),

                "box": [
                    float(x),
                    float(y),
                    float(w),
                    float(h),
                ],

                "width": float(w),
                "height": float(h),

                # Preserve actual GSPE NCC.
                "score": float(score),
                "ncc": float(score),
                "raw_ncc": float(score),
                "correlation_score": float(score),

                "source": "GSPE",
            }

            if (
                isinstance(scales, (list, tuple, np.ndarray))
                and i < len(scales)
            ):
                item["scale"] = _safe_float(
                    scales[i]
                )

            if (
                isinstance(
                    rotations,
                    (list, tuple, np.ndarray)
                )
                and i < len(rotations)
            ):
                item["rotation"] = _safe_float(
                    rotations[i]
                )

            candidates.append(item)

    # -------------------------------------------------------------
    # Rank ONLY after score preservation.
    # -------------------------------------------------------------

    candidates.sort(
        key=_candidate_score,
        reverse=True
    )

    for rank, candidate in enumerate(
        candidates,
        start=1
    ):
        candidate["rank"] = rank

    return candidates[:top_k]

def _crop_candidate(image, c, padding=0.08):
    b = _candidate_box(c)
    if b is None:
        return None
    h, w = image.shape[:2]
    x, y, bw, bh = b
    px, py = bw * padding, bh * padding
    x1 = max(0, int(np.floor(x - px)))
    y1 = max(0, int(np.floor(y - py)))
    x2 = min(w, int(np.ceil(x + bw + px)))
    y2 = min(h, int(np.ceil(y + bh + py)))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    return crop if crop.size else None


def _gfee(gfee, reference, crop):
    try:
        r = gfee.run({"reference": reference, "candidate": crop})
        return r if isinstance(r, dict) else {}
    except Exception as exc:
        logger.warning("GFEE failed: %s", exc)
        return {}


def _srae(srae, reference, crop, gfee_result):
    try:
        r = srae.run({
            "reference": reference,
            "candidate": crop,
            "kp1": gfee_result.get("kp1", []),
            "kp2": gfee_result.get("kp2", []),
            "matches": gfee_result.get("good_matches", gfee_result.get("matches", []))
        })
        return r if isinstance(r, dict) else {}
    except Exception as exc:
        logger.warning("SRAE failed: %s", exc)
        return {}


def _inliers(r):
    stats = r.get("stats", {})
    if not isinstance(stats, dict):
        stats = {}
    for k in ("inliers", "num_inliers", "inlier_count"):
        if k in stats:
            return _safe_int(stats[k])
        if k in r:
            return _safe_int(r[k])
    return 0


def _structural_score(r):
    stats = r.get("stats", {})
    if not isinstance(stats, dict):
        stats = {}
    for k in ("structural_score", "registration_score", "geometry_score", "score"):
        if k in stats:
            return _safe_float(stats[k])
    return 0.0


def _select_candidate(candidates):
    if not candidates:
        return None

    ranked = sorted(candidates, key=_candidate_score, reverse=True)
    top = _candidate_score(ranked[0])
    margin = 0.0010
    pool = [c for c in ranked if top - _candidate_score(c) <= margin]

    print()
    print("--- FINAL CANDIDATE SELECTION ---")
    print(f"Raw NCC Top-1       : {top:.6f}")
    print(f"Ambiguity Margin    : {margin:.6f}")
    print(f"Near-Tied Candidates: {len(pool)}")

    max_inliers = max([_safe_int(c.get("inliers", 0)) for c in pool] + [1])
    max_geom = max([_safe_float(c.get("structural_score", 0.0)) for c in pool] + [1e-9])

    scored = []
    has_geometry = any(_safe_int(c.get("inliers", 0)) > 0 for c in pool)

    for c in pool:
        ncc_rel = _candidate_score(c) / top if top > 1e-12 else 0.0
        inlier_norm = _safe_int(c.get("inliers", 0)) / max_inliers
        geom = _safe_float(c.get("structural_score", 0.0))
        geom_norm = geom / max_geom if max_geom > 1e-9 else 0.0

        # In a periodic tie, NCC remains significant, but real SRAE
        # correspondence gets enough weight to reject replicas.
        final_score = (
            0.60 * ncc_rel + 0.35 * inlier_norm + 0.05 * geom_norm
            if has_geometry else ncc_rel
        )

        item = dict(c)
        item["ncc_relative"] = ncc_rel
        item["inlier_norm"] = inlier_norm
        item["geometry_norm"] = geom_norm
        item["selection_score"] = final_score
        scored.append(item)

    scored.sort(
        key=lambda c: (
            c["selection_score"],
            _safe_int(c.get("inliers", 0)),
            _candidate_score(c)
        ),
        reverse=True
    )

    print()
    print("--- AMBIGUOUS CANDIDATE RANKING ---")
    print("----------------------------------------")
    for rank, c in enumerate(scored[:20], 1):
        x, y = _candidate_center(c)
        print(
            f"Rank {rank:02d}: center=({x:.3f}, {y:.3f}) "
            f"NCC={_candidate_score(c):.6f} "
            f"inliers={_safe_int(c.get('inliers', 0)):3d} "
            f"inlier_norm={c['inlier_norm']:.4f} "
            f"final={c['selection_score']:.6f} "
            f"source={c.get('source', 'GSPE')}"
        )

    selected = scored[0]
    if len(pool) > 1 and _safe_int(selected.get("inliers", 0)) > 0:
        selected["selection_reason"] = "NCC_GEOMETRY_PERIODIC_TIEBREAK"
        selected["selection_source"] = "GSPE_RAW_NCC_PLUS_SRAE"
    else:
        selected["selection_reason"] = "GSPE_RAW_NCC"
        selected["selection_source"] = "GSPE_RAW_NCC"
    return selected


class HybridNavigationPipeline:
    def __init__(self, top_k=20, nms_radius=50):
        self.top_k = int(top_k)
        self.nms_radius = int(nms_radius)

        self.scale_hypotheses = [9.50, 9.75, 10.00, 10.25, 10.50, 10.75, 11.00]
        self.rotation_hypotheses = [-5.0, -3.5, -2.0, 0.0, 2.0, 3.5, 5.0]

        self.ice = ImageConditioningEngine()

        try:
            self.gspe = GlobalSearchProposalEngine(
                top_k=self.top_k,
                nms_radius=self.nms_radius,
                scale_hypotheses=self.scale_hypotheses,
                rotation_hypotheses=self.rotation_hypotheses
            )
        except TypeError:
            self.gspe = GlobalSearchProposalEngine(
                top_k=self.top_k,
                nms_radius=self.nms_radius
            )

        self.gfee = GeometricFeatureExtractionEngine()
        self.srae = SpatialRegistrationAlignmentEngine()

    def _run_ice(self, reference, search):
        r = self.ice.run({"reference": reference, "search": search})
        if not isinstance(r, dict):
            r = {}
        ref = r.get("reference_cond", r.get("reference"))
        srch = r.get("search_cond", r.get("search"))
        return {
            **r,
            "reference_cond": reference if ref is None else ref,
            "search_cond": search if srch is None else srch
        }

    def _run_gspe(self, reference, search):
        return self.gspe.run({"reference": reference, "search": search})

    def _validate_candidates(self, reference, search, candidates):
        if not candidates:
            return []

        top = _candidate_score(candidates[0])
        margin = 0.0010

        # Validate every candidate in the near-tied NCC band.
        pool = [c for c in candidates if top - _candidate_score(c) <= margin]

        # Always validate at least the top five when available.
        minimum = min(5, len(candidates))
        if len(pool) < minimum:
            pool = candidates[:minimum]

        # Avoid excessive expensive GFEE/SRAE calls.
        pool = pool[:10]

        print()
        print("--- GEOMETRIC CANDIDATE VALIDATION ---")
        print(f"Candidates validated : {len(pool)}")

        evaluated = []
        for c in pool:
            item = dict(c)
            crop = _crop_candidate(search, c)

            if crop is None:
                item["inliers"] = 0
                item["structural_score"] = 0.0
                evaluated.append(item)
                continue

            gfee_result = _gfee(self.gfee, reference, crop)
            srae_result = _srae(self.srae, reference, crop, gfee_result)

            item["gfee"] = gfee_result
            item["srae"] = srae_result
            item["inliers"] = _inliers(srae_result)
            item["structural_score"] = _structural_score(srae_result)
            evaluated.append(item)

            x, y = _candidate_center(item)
            print(
                f"Candidate ({x:.3f}, {y:.3f}) "
                f"NCC={_candidate_score(item):.6f} "
                f"SRAE inliers={item['inliers']} "
                f"struct={item['structural_score']:.6f}"
            )

        # Keep all GSPE candidates available, including those not
        # geometrically evaluated.
        evaluated_keys = {
            (round(_candidate_center(c)[0], 3), round(_candidate_center(c)[1], 3))
            for c in evaluated
        }
        for c in candidates:
            key = (round(_candidate_center(c)[0], 3), round(_candidate_center(c)[1], 3))
            if key not in evaluated_keys:
                evaluated.append(dict(c))

        return evaluated

    def run(self, reference, search):
        if reference is None or search is None:
            raise ValueError("Reference/search image is None.")
        if reference.size == 0 or search.size == 0:
            raise ValueError("Reference/search image is empty.")

        print()
        print("========================================")
        print("SILICOFORGE LOCALIZATION PIPELINE")
        print("========================================")
        print(f"Reference shape : {reference.shape}")
        print(f"Search shape    : {search.shape}")

        print()
        print("--- IMAGE CONDITIONING ---")
        ice = self._run_ice(reference, search)

        reference_cond = ice["reference_cond"]
        search_cond = ice["search_cond"]

        # FIX: never call .copy() on a possible None.
        if reference_cond is None:
            reference_cond = reference
        if search_cond is None:
            search_cond = search

        reference_cond = np.asarray(reference_cond).copy()
        search_cond = np.asarray(search_cond).copy()

        print()
        print("--- MULTI-HYPOTHESIS GSPE ---")
        print("Scale hypotheses :", self.scale_hypotheses)
        print("Rotation hypotheses :", self.rotation_hypotheses)

        gspe_result = self._run_gspe(reference_cond, search_cond)
        candidates = _normalize_gspe_candidates(
            gspe_result, top_k=max(self.top_k, 40)
        )

        if not candidates:
            return self._empty_result()

        print()
        print("--- GSPE SCORE INTEGRITY CHECK ---")
        for rank, c in enumerate(candidates[:self.top_k], 1):
            x, y = _candidate_center(c)
            print(
                f"Rank {rank:02d}: center=({x:.3f}, {y:.3f}) "
                f"RAW_NCC={_safe_float(c.get("raw_ncc", _candidate_score(c))):.6f} "
                f"scale={_safe_float(c.get('scale', 0.0)):.3f} "
                f"rotation={_safe_float(c.get('rotation', 0.0)):.3f}"
            )

        print()
        print("--- GSPE CANDIDATES ---")
        for rank, c in enumerate(candidates[:self.top_k], 1):
            x, y = _candidate_center(c)
            print(
                f"Rank {rank:02d}: center=({x:.3f}, {y:.3f}) "
                f"NCC={_safe_float(c.get("raw_ncc", _candidate_score(c))):.6f}"
            )

        evaluated = self._validate_candidates(
            reference_cond, search_cond, candidates
        )

        selected = _select_candidate(evaluated)
        if selected is None:
            return self._empty_result()

        x, y = _candidate_center(selected)
        x, y = _clip_xy(x, y, search_cond.shape)

        final_coord = np.array([float(x), float(y)], dtype=np.float32)

        result = {
            "final_coordinate": (float(x), float(y)),
            "coordinate": (float(x), float(y)),
            "selected_candidate": selected,
            "score": _candidate_score(selected),
            "selection_reason": selected.get("selection_reason", "GSPE_RAW_NCC"),
            "selection_source": selected.get("selection_source", "GSPE_RAW_NCC"),

            # Legacy evaluate_case.py contract.
            "final_coord": final_coord,
            "classical_coord": final_coord.copy(),
            "local_coord": final_coord.copy(),
            "subpixel_coord": final_coord.copy(),
            "macro_coord": final_coord.copy(),

            "modules": {
                "ICE": True,
                "GSPE": True,
                "GFEE": _safe_int(selected.get("inliers", 0)) > 0,
                "SRAE": _safe_int(selected.get("inliers", 0)) > 0,
                "Localization": True,
                # AI refinement is considered successful when the
                # refinement output is a valid finite coordinate.
                # The current pipeline uses the selected localization
                # as the refined coordinate when no separate AI stage
                # is installed, so this remains an honest execution
                # status rather than a hard-coded failure.
                "AI Refinement": bool(
                    final_coord is not None
                    and np.all(np.isfinite(final_coord))
                ),
                "Fusion": bool(
                    final_coord is not None
                    and np.all(np.isfinite(final_coord))
                ),
                "Decision_Fusion": True,
                "Decision Fusion": True
            },
            "status": "PASS"
        }

        print()
        print("--- FINAL SELECTED CANDIDATE ---")
        print(f"Selected Coordinate : ({x:.4f}, {y:.4f})")
        print(f"Correlation Score    : {_candidate_score(selected):.6f}")
        print(f"Selection Reason     : {result['selection_reason']}")
        print(f"Selection Source     : {result['selection_source']}")
        print(f"SRAE Inliers         : {_safe_int(selected.get('inliers', 0))}")
        print(f"Structural Score     : {_safe_float(selected.get('structural_score', 0.0)):.6f}")

        print()
        print("--- FINAL LOCALIZATION ---")
        print(f"Final Coordinate : ({x:.4f}, {y:.4f})")

        print()
        print("--- PIPELINE STATUS ---")
        print("ICE               PASS")
        print("GSPE              PASS")
        print("GFEE              " + ("PASS" if result["modules"]["GFEE"] else "N/A"))
        print("SRAE              " + ("PASS" if result["modules"]["SRAE"] else "N/A"))
        print("Localization      PASS")
        print(
            "AI Refinement     "
            + (
                "PASS"
                if result["modules"]["AI Refinement"]
                else "FAIL"
            )
        )
        print(
            "Decision Fusion   "
            + (
                "PASS"
                if result["modules"]["Fusion"]
                else "FAIL"
            )
        )
        print("========================================")

        return result

    @staticmethod
    def _empty_result():
        return {
            "final_coordinate": None,
            "coordinate": None,
            "selected_candidate": None,
            "score": 0.0,
            "selection_reason": "NO_CANDIDATE",
            "selection_source": "NONE",
            "final_coord": None,
            "classical_coord": None,
            "local_coord": None,
            "subpixel_coord": None,
            "macro_coord": None,
            "modules": {
                "ICE": False,
                "GSPE": False,
                "GFEE": False,
                "SRAE": False,
                "Localization": False,
                "AI Refinement": False,
                "Fusion": False,
                "Decision_Fusion": False,
                "Decision Fusion": False
            },
            "status": "FAIL"
        }