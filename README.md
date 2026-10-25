# FuDensityNet

**Density-aware occlusion assessment from RGB and LiDAR geometry**

FuDensityNet investigates geometric cues that can help identify object proposals affected by challenging visibility conditions. The present repository contains the proposal-level Occlusion Rate (OR) implementation, KITTI evaluation utilities, and visual results presented at the University of Antwerp.

The pipeline projects LiDAR points into the camera image, collects the points within each proposed object region, separates near and far depth layers, measures foreground density at three voxel resolutions, and combines the resulting descriptors into an OR score.

![Proposal-level analysis](figures/002050_slide5_combined_proposal_level_analysis_v2.png)

## Method

1. Project the LiDAR point cloud into the RGB image using the camera calibration.
2. Collect projected points within each 2D object proposal (with a 5% box expansion).
3. Sort the selected points by depth and look for the largest depth discontinuity. A split is accepted for a gap of at least 3 m and at least two points per layer.
4. Compute voxel densities from the near layer at **0.25 m, 0.50 m and 1.00 m**.
5. Apply the calibration-fitted normalization to the density and depth descriptors, and combine them into a proposal-level score.

The implemented combination is

```text
OR_density = 0.50 D_small + 0.30 D_medium + 0.20 D_large
OR = 0.70 OR_density + 0.20 D_gap + 0.10 D_balance
```

`D_gap` is calculated from the largest raw depth gap using the calibration-set 5th and 95th percentiles. `D_balance` is the near/far minority proportion scaled to 0–100 and capped at a minority fraction of 0.30. Voxel densities are also adjusted by squared median depth and the balance gate before percentile normalization. The exact implementation is in `src/fudensitynet/or_score.py`.

![Depth layers](figures/002050_slide6_sorted_depth_gap_annotated.png)

## Results

In the earlier KITTI oracle-proposal experiments, occlusion annotation 0 was treated as visible and annotations 1 and 2 as occluded. The reported five-split results were:

| Metric | Mean ± standard deviation |
| --- | ---: |
| ROC AUC | 0.647 ± 0.015 |
| Average precision | 0.631 ± 0.018 |
| Balanced accuracy | 0.614 ± 0.016 |
| Calibrated OR threshold | 12 |

These values describe **discrimination of occlusion annotations within supplied proposals**, not object detection accuracy. The values are recorded from earlier experiments; the included test suite checks the numerical methods and does not by itself rerun the full KITTI study.

See [results/README.md](results/README.md) for the experiment setting, additional observations, and available figures.

## Repository contents

```text
src/fudensitynet/      OR computation, KITTI interfaces and scoring utilities
scripts/               Dataset evaluation, candidate-box scoring and plots
notebooks/             Compact example of the implemented score
figures/               KITTI figures used during the research presentation
results/               Recorded metrics and figure index
tests/                 Automated tests
```

## Installation

Python 3.10 or newer is recommended. From a terminal in the repository directory:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

For a KITTI experiment, provide the standard KITTI object-training folders (`image_2`, `velodyne`, `calib`, `label_2`):

```bash
python scripts/evaluate_kitti.py --kitti-training /path/to/kitti_dataset/training --max-frames 500 --seed 42 --output-dir runs/seed42
```

To score a set of externally supplied 2D proposals with a previously fitted calibration:

```bash
python scripts/score_proposals.py --image image.png --velodyne scan.bin --calib calib.txt --boxes examples/proposals.json --calibration runs/seed42/calibration.json
```

The input bounding boxes are provided externally; the current code does not train or execute an object detector. No KITTI raw data or pretrained weights are distributed in this repository.

## Next steps

The next development stage is integration with a 2D detector and a trained RGB–LiDAR fusion network. OR-guided routing and distance-aware prioritization are research extensions; detection-level comparisons are separate from the proposal-level OR evaluation presented here.

## Research presentation

The figures in this repository are associated with the Antwerp research presentation. The complete architecture presented there provides context for the intended FuDensityNet system; this repository implements the geometric OR component.
