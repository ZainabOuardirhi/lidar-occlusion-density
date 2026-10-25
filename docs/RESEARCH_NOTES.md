# Evidence, provenance and discrepancies

Source of the core implemented functions: uploaded `implementation(1).ipynb` (latest reconstruction), particularly the functions `select_lidar_points_in_2d_box`, `split_depth_layers_from_proposal`, `compute_raw_proposal_density_or_features`, `fit_proposal_or_calibration_params`, `score_proposal_density_or_with_params` and frame-disjoint evaluation. The functions are extracted into importable modules with minimal dependency wiring. The runner's KITTI file traversal was reimplemented to remove hardcoded HumaLearn paths and to avoid OpenCV dependency.

The Antwerp slides are an **explanatory presentation**, not a substitute for source code or reproducibility artifacts. Certain displayed depth-layer formulas are simplified and do not exactly match the current implementation: the implementation normalizes **raw maximum depth gap with calibration quantiles** and uses a **saturating minority-layer balance**. The code is authoritative for this release.

Prior conversation reported five calibration/test splits, seeds 42–46, with approximate AUC 0.647 ± 0.015, AP 0.631 ± 0.018, and balanced accuracy 0.614 ± 0.016, threshold 12. **These are historical notebook run claims, not reproduced by this package in this environment because KITTI raw files were not supplied.** Run the included evaluation script on complete KITTI training data and verify the split/sample assumptions before presenting as freshly reproduced.

The exploratory distance-aware multiplier and feature-weight adaptation are **not** deployed in this package because a held-out detection or calibration improvement has not been demonstrated. Any boxes shown on figures annotated with KITTI labels were oracle proposal illustrations, not autonomous detection.

Do not accidentally describe this phase as object detection or end-to-end OR-gated YOLOv8. No neural network is trained by this package.
