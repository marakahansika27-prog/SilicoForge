import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.integration.decision_fusion import DecisionFusionEngine

def run_test(case_name, classical, ai, conf, expected, engine):
    inputs = {
        'classical_coord': classical,
        'ai_coord': ai,
        'confidence': conf
    }
    print(f"\nEvaluating {case_name}...")
    res = engine.run(inputs)
    actual = res['decision']
    if actual == expected:
        print(f"[PASS] {case_name}")
        return True
    else:
        print(f"[FAIL] {case_name} | Expected {expected}, got {actual}")
        return False

def main():
    engine = DecisionFusionEngine(confidence_threshold=0.90, residual_deadband=0.10)
    passed = 0
    total = 8
    
    classical = np.array([100.0, 100.0])
    
    # CASE 1
    ai_1 = classical + np.array([0.0622, 0.0])
    passed += run_test("CASE 1: conf=0.99, res=0.0622", classical, ai_1, 0.99, "CLASSICAL_FALLBACK", engine)
    
    # CASE 2
    ai_2 = classical + np.array([0.50, 0.0])
    passed += run_test("CASE 2: conf=0.99, res=0.50", classical, ai_2, 0.99, "AI_REFINED", engine)
    
    # CASE 3
    ai_3 = classical + np.array([0.50, 0.0])
    passed += run_test("CASE 3: conf=0.50, res=0.50", classical, ai_3, 0.50, "CLASSICAL_FALLBACK", engine)
    
    # CASE 4
    ai_4 = classical + np.array([0.50, 0.0])
    passed += run_test("CASE 4: conf=NaN", classical, ai_4, float('nan'), "CLASSICAL_FALLBACK", engine)
    
    # CASE 5
    ai_5 = classical + np.array([float('nan'), 0.0])
    passed += run_test("CASE 5: res contains NaN", classical, ai_5, 0.99, "CLASSICAL_FALLBACK", engine)
    
    # CASE 6
    ai_6 = classical + np.array([float('inf'), 0.0])
    passed += run_test("CASE 6: res contains Inf", classical, ai_6, 0.99, "CLASSICAL_FALLBACK", engine)
    
    # CASE 7
    ai_7 = classical + np.array([0.1000, 0.0])
    passed += run_test("CASE 7: res exactly 0.1000", classical, ai_7, 0.99, "CLASSICAL_FALLBACK", engine)
    
    # CASE 8
    ai_8 = classical + np.array([0.1001, 0.0])
    passed += run_test("CASE 8: res slightly above deadband", classical, ai_8, 0.99, "AI_REFINED", engine)
    
    print(f"\nScore: {passed}/{total}")
    if passed == total:
        print("\n[DECISION FUSION DEADBAND VERIFIED]")
        sys.exit(0)
    else:
        print("\nTests Failed.")
        sys.exit(1)

if __name__ == '__main__':
    main()
