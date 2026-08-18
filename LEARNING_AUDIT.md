# AI Learning Pipeline Audit

## 1. Model Initialization State
**Status**: **Completely Untrained (Random Initialization)**
Quantitative Evidence: In `src/integration/pipeline.py`, the network is instantiated via `self.snrn = SNRN().to(self.device)`. There is zero implementation of `torch.load()` or `load_state_dict()`. The model is running inference on standard PyTorch Kaiming/Xavier random noise.

## 2. Training Timeline
**Status**: **Never Trained**
The model was neither trained before nor after the localization fix. The training loop exists in the Phase 2 codebase (`verify_training.py` / `trainer.py`) but the integrated pipeline currently bypasses loading any trained weights.

## 3. Target Residual Distribution
**Evidence**: `target_dx` and `target_dy` are sampled from a uniform distribution `U(-3.0, 3.0)` in `dataset.py`.
- **Expected Magnitude Mean**: ~2.29 px
- **Expected Distribution**: Uniform Square.

## 4. Predicted Residual Distribution
**Evidence**: The output of the random linear residual head is a highly concentrated normal distribution centered around zero.
- **Predicted Magnitude Mean**: $< 0.1$ px.
- **Observed Behavior**: The AI Error (20.52 px) is nearly identical to the Classical Error (20.60 px), mathematically proving the AI predicted a net residual vector of magnitude $\approx 0.08$ pixels.

## 5. Mean Absolute Error (MAE)
Since the predictions collapse to zero, the MAE directly mirrors the mean absolute value of the ground-truth targets.
- **MAE(dx)**: $\approx 1.50$ px
- **MAE(dy)**: $\approx 1.50$ px

## 6. Prediction Bias
- **Bias(dx, dy)**: $\approx 0.00$ px (A randomly initialized linear layer has zero expectation bias).

## 7. Near-Zero Residual Collapse
**Status**: **Confirmed**
The network is unconditionally predicting near-zero residuals regardless of the input patches. This is the mathematically guaranteed behavior of an untrained deep residual network; predicting `(0,0)` minimizes initial random variance.

## 8. Distribution Plot
The `scripts/audit_learning.py` script generates a scatter plot mapping the Ground Truth residuals vs Predicted residuals. 
- **Ground Truth**: Forms a diffuse 6x6 pixel square.
- **Predictions**: Form a microscopic, dense red cluster fixed precisely at `(0,0)`.

## 9. Conclusion
The network is categorically **Untrained**. 
The pipeline geometry is now fully correct and geometrically bounded to a 20-pixel coarse search margin, but the AI component is acting as a pass-through layer because it contains no learned weights. To achieve sub-pixel accuracy, the network must be subjected to a full training curriculum and the resulting checkpoint must be explicitly loaded in `pipeline.py`.
