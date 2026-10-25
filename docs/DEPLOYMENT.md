# GitHub deployment and application integration

## 1. Before uploading

The repository intentionally **excludes KITTI raw data, trained weights, large notebook outputs and sensitive personal paths**. Do not upload institutional datasets without permission.

## 2. Publish using GitHub

1. Sign into GitHub and create an **empty** repository (e.g. `FuDensityNet`). Do not initialize it with a README.
2. Extract this ZIP and open a terminal in `FuDensityNet_Phase1/`.
3. Run:

```bash
git init
git add .
git commit -m "Initial research release: proposal-level occlusion assessment"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/FuDensityNet.git
git push -u origin main
```

Authenticate using Git Credential Manager or your normal GitHub credential workflow. Replace YOUR_ACCOUNT; do not embed a password or token in the URL.

## 3. Use from an application

Install with `pip install -e .`. Read calibration from `calibration.json`. To score proposed boxes in a frame use:

```python
from fudensitynet.pipeline import score_frame_proposals,frame_fusion_required
scored = score_frame_proposals(lidar_xyzi, calibration_matrices, image.shape, boxes, params)
route_to_fusion = frame_fusion_required(scored, threshold)
```

`boxes` must be supplied by your own detector. When `route_to_fusion` is True, **your application must implement its fusion branch separately**. Phase 1 does not provide either branch. If adapting to other sensor rigs, update the coordinate transformations and calibrate using representative annotated data.

## 4. Production-readiness checklist

- Add a real proposal generator and evaluate on held-out detector outputs.
- Implement and train RGB-only and RGB–LiDAR fusion branches.
- Test end-to-end detection precision/recall/mAP, cost and latency per occlusion level.
- Calibrate and evaluate against independent sequences and hardware.
- Add error handling for missing / stale / unsynchronized sensors and invalid boxes.
- Profile execution, set operational thresholds, and document failure modes.

This is a research SDK, not a hosted API or deployed service.
