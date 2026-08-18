import os
import sys
import numpy as np
import traceback

class PipelineValidationError(Exception):
    pass

def validate_inputs(ref, search):
    if ref is None or search is None:
        raise PipelineValidationError("Input images cannot be None.")
    if not isinstance(ref, np.ndarray) or not isinstance(search, np.ndarray):
        raise PipelineValidationError("Input images must be NumPy arrays.")
    if ref.size == 0 or search.size == 0:
        raise PipelineValidationError("Input images cannot be empty.")
    if ref.shape[0] == 0 or ref.shape[1] == 0 or search.shape[0] == 0 or search.shape[1] == 0:
        raise PipelineValidationError("Input dimensions must be non-zero.")
    if np.isnan(ref).any() or np.isnan(search).any():
        raise PipelineValidationError("Input images cannot contain NaN.")
    if np.isinf(ref).any() or np.isinf(search).any():
        raise PipelineValidationError("Input images cannot contain Inf.")
    if np.var(ref) < 1e-6:
        raise PipelineValidationError("Reference image has near-zero variance.")

def validate_template(ref_scaled, search_shape):
    if ref_scaled is None or ref_scaled.size == 0:
        raise PipelineValidationError("Scaled template is invalid or empty.")
    if np.isnan(ref_scaled).any() or np.isinf(ref_scaled).any():
        raise PipelineValidationError("Scaled template contains NaN/Inf.")
    if np.var(ref_scaled) < 1e-6:
        raise PipelineValidationError("Scaled template has near-zero variance.")
    if ref_scaled.shape[0] > search_shape[0] or ref_scaled.shape[1] > search_shape[1]:
        raise PipelineValidationError("Scaled template is larger than the search image.")

def validate_response_map(res):
    if res is None or res.size == 0:
        raise PipelineValidationError("Response map is invalid or empty.")
    if np.isnan(res).any() or np.isinf(res).any():
        raise PipelineValidationError("Response map contains NaN/Inf.")

def validate_subpixel(dx, dy):
    if np.isnan(dx) or np.isinf(dx) or abs(dx) > 0.5:
        dx = 0.0
    if np.isnan(dy) or np.isinf(dy) or abs(dy) > 0.5:
        dy = 0.0
    return dx, dy

def validate_final_coordinate(x, y, search_shape):
    if np.isnan(x) or np.isinf(x) or np.isnan(y) or np.isinf(y):
        raise PipelineValidationError(f"Final coordinate is invalid: ({x}, {y})")
    if x < 0 or x >= search_shape[1] or y < 0 or y >= search_shape[0]:
        raise PipelineValidationError(f"Final coordinate is out of bounds: ({x}, {y})")

def run_negative_tests():
    tests = [
        ("None reference", None, np.ones((10,10))),
        ("None search", np.ones((10,10)), None),
        ("Empty array", np.array([]), np.ones((10,10))),
        ("Zero-sized array", np.ones((0, 10)), np.ones((10,10))),
        ("NaN-containing image", np.array([[np.nan, 1]]), np.ones((10,10))),
        ("Inf-containing image", np.array([[np.inf, 1]]), np.ones((10,10))),
        ("Zero-variance template", np.ones((10,10)), np.ones((20,20))),
        ("Template > search", np.ones((30,30)), np.ones((20,20)))
    ]
    
    print("--- NEGATIVE TESTS ---")
    for name, ref, search in tests:
        passed = False
        try:
            validate_inputs(ref, search)
            if name == "Template > search":
                validate_template(ref, search.shape)
            print(f"[{name}] - FAILED (Did not raise exception)")
        except PipelineValidationError as e:
            print(f"[{name}] - PASSED: {str(e)}")
            passed = True
        except Exception as e:
            print(f"[{name}] - FAILED (Unexpected exception: {e})")

if __name__ == "__main__":
    run_negative_tests()
