# References

1. **Normalized Cross-Correlation and Template Matching**:
   - Briechle, K., & Hanebeck, U. D. (2001). Template matching using fast normalized cross correlation. *Proceedings of SPIE - The International Society for Optical Engineering*.
   - Justifies the use of the Global Search Proposal Engine (GSPE) and NCC for initial coarse-level alignment and bounding box identification.

2. **Subpixel Registration and Phase Correlation**:
   - Foroosh, H., Zerubia, J. B., & Berthod, M. (2002). Extension of phase correlation to subpixel registration. *IEEE Transactions on Image Processing*, 11(3), 188-200.
   - Grounding for classical interpolation strategies inside the spatial registration alignment engines.

3. **Noise and Structural Perturbation Modeling (ICE)**:
   - Foi, A., Trimeche, M., Katkovnik, V., & Egiazarian, K. (2008). Practical Poissonian-Gaussian noise modeling and fitting for single-image raw-data. *IEEE Transactions on Image Processing*, 17(10), 1737-1754.
   - Informs the Image Conditioning Engine (ICE) and synthetic dataset generation protocols for realistic semiconductor noise characteristics.

4. **Residual Network Refinement (SNRN)**:
   - He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 770-778).
   - Underpins the architecture of the Subpixel Navigation Refinement Network (SNRN) designed to predict residual affine warps.

5. **Decision Fusion and Confidence Gating**:
   - Kendall, A., & Gal, Y. (2017). What uncertainties do we need in bayesian deep learning for computer vision? *Advances in neural information processing systems*, 30.
   - Inspired the confidence-aware Decision Fusion Engine, utilizing spatial structural consensus to dynamically override neural outputs during uncertainty or catastrophic failure events.
