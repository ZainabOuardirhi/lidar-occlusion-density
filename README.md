# FuDensityNet: Phase 1: Density-Aware Occlusion Assessment

**Status: research reference implementation, not a trained multimodal detector or production service.**

This package isolates the **proposal-level Occlusion Rate (OR)** computation developed in the FuDensityNet research notebook. Its purpose is to score geometric complexity/occlusion-prone regions using projected LiDAR inside **supplied 2D proposal boxes**. In KITTI evaluation those boxes are *ground-truth oracle proposals*. At application time they must come from a detector or another proposal provider, which is **not included**.

## Install locally

Python 3.10+ (PowerShell):

```powershell
cd FuDensityNet_Phase1
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
```

Ubuntu/WSL:

```bash
cd FuDensityNet_Phase1
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

## Run a reproducible KITTI experiment

Download KITTI **object detection training** data independently under its license. Supply a directory holding `image_2/`, `velodyne/`, `calib/`, `label_2/`. No KITTI images or LiDAR are redistributed here.

```bash
python scripts/evaluate_kitti.py --kitti-training /path/to/kitti_dataset/training --max-frames 500 --seed 42 --output-dir runs/seed42
```

Output: calibration JSON (including a calibrated threshold), test metrics JSON, and proposal-level test scores CSV. Train/calibration and test proposals are split **by frame ID**, and all robust percentiles are fit on calibration frames only. This is **OR binary-occlusion ranking evaluation**, not detection mAP.

## Score candidate boxes for an application

Once a calibration file is available, supply RGB image size, LiDAR scan, KITTI-format calibration and proposed 2D boxes as a JSON array (`[[x1,y1,x2,y2], ...]`):

```bash
python scripts/score_proposals.py --image image.png --velodyne scan.bin --calib calib.txt --boxes proposals.json --calibration runs/seed42/calibration.json
```

This produces per-proposal OR and a **routing recommendation only**. It neither locates objects nor executes a multimodal neural network. Check time synchronization and geometric calibration before use on new camera/LiDAR hardware.

## Actual implemented OR formula

For projected LiDAR points inside a 2D box expanded by 5%, split at the **largest adjacent camera-depth gap**, if gap >= 3 m and >=2 points per layer. The near layer is used for voxel densities for voxel sizes 0.25 / 0.50 / 1.00 m. Density for voxel size s is `N_near/(occupied_voxels*s**3)`. Let `balance=min(N_near,N_far)/(N_near+N_far)` and `Dbalance=100*clip(balance/0.30,0,1)`. Correct density by `(median_camera_depth**2) * (Dbalance/100)` and transform each density and **raw max depth gap** using separate calibration-fitted 5th/95th percentile clipping onto [0,100]. Then:

`Ddensity = 0.50*Dsmall + 0.30*Dmedium + 0.20*Dlarge`

`OR = 0.70*Ddensity + 0.20*Dgap + 0.10*Dbalance`

When no depth split qualifies, `OR=0`. This is a geometric heuristic with data-dependent calibration, **not a posterior probability**.

See `docs/RESEARCH_NOTES.md` and `docs/DEPLOYMENT.md`.
