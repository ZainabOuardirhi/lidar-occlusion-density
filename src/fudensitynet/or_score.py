"""Density-aware proposal-level occlusion assessment."""
import numpy as np
import pandas as pd
from .kitti import project_lidar_to_image

def select_lidar_points_in_2d_box(
    points,
    calib,
    image_shape,
    bbox,
    expansion=0.05,
):
    """
    Select projected LiDAR points inside a 2D proposal box.

    The box can come from:
    - KITTI labels during validation,
    - YOLO predictions during inference.

    Occlusion label is not used.
    """
    height, width = image_shape[:2]

    x1, y1, x2, y2 = bbox

    box_w = x2 - x1
    box_h = y2 - y1

    x_margin = expansion * box_w
    y_margin = expansion * box_h

    x1e = max(0.0, x1 - x_margin)
    y1e = max(0.0, y1 - y_margin)
    x2e = min(width - 1.0, x2 + x_margin)
    y2e = min(height - 1.0, y2 + y_margin)

    xyz = points[:, :3].astype(np.float32)

    uv, depth, valid = project_lidar_to_image(
        xyz,
        calib,
    )

    inside = (
        valid
        & np.isfinite(uv[:, 0])
        & np.isfinite(uv[:, 1])
        & np.isfinite(depth)
        & (uv[:, 0] >= x1e)
        & (uv[:, 0] <= x2e)
        & (uv[:, 1] >= y1e)
        & (uv[:, 1] <= y2e)
        & (depth > 0)
    )

    return {
        "xyz": xyz[inside],
        "uv": uv[inside],
        "depth": depth[inside],
        "expanded_bbox": [x1e, y1e, x2e, y2e],
    }

def split_depth_layers_from_proposal(
    depths,
    min_depth_gap=3.0,
    min_points_per_layer=2,
):
    """
    Split LiDAR points inside a proposal into near and far depth layers.

    No object depth and no occlusion label are used.

    If a strong depth gap exists:
        near layer = possible foreground occluder
        far layer = possible target/background layer
    """
    depths = np.asarray(depths, dtype=np.float32)

    if len(depths) < 2 * min_points_per_layer:
        return None

    sorted_depths = np.sort(depths)
    gaps = np.diff(sorted_depths)

    if len(gaps) == 0:
        return None

    gap_idx = int(np.argmax(gaps))
    max_gap = float(gaps[gap_idx])

    if max_gap < min_depth_gap:
        return None

    split_depth = float(
        0.5 * (
            sorted_depths[gap_idx]
            + sorted_depths[gap_idx + 1]
        )
    )

    near_mask = depths <= split_depth
    far_mask = depths > split_depth

    if near_mask.sum() < min_points_per_layer:
        return None

    if far_mask.sum() < min_points_per_layer:
        return None

    return {
        "split_depth": split_depth,
        "max_depth_gap": max_gap,
        "near_mask": near_mask,
        "far_mask": far_mask,
        "n_near": int(near_mask.sum()),
        "n_far": int(far_mask.sum()),
    }

def compute_voxel_density_for_points(
    xyz,
    voxel_size,
):
    """
    Compute density from points inside a proposal.
    """
    if xyz is None or len(xyz) == 0:
        return 0.0, 0.0, 0.0

    origin = xyz.min(axis=0)

    voxel_idx = np.floor(
        (xyz - origin) / voxel_size
    ).astype(np.int32)

    _, counts = np.unique(
        voxel_idx,
        axis=0,
        return_counts=True,
    )

    occupied_voxels = len(counts)

    occupied_volume = max(
        occupied_voxels * (voxel_size ** 3),
        1e-6,
    )

    density = len(xyz) / occupied_volume

    return float(density), float(occupied_voxels), float(counts.max())

def compute_raw_proposal_density_or_features(
    points,
    calib,
    image_shape,
    bbox,
    expansion=0.05,
    voxel_sizes=(0.25, 0.50, 1.00),
    min_depth_gap=3.0,
):
    """
    Compute raw density-aware OR features inside a proposal.

    No occlusion label is used.

    Main idea:
        if there is a dense foreground layer inside the proposal,
        and it is separated from farther points,
        then the region is likely occlusion-prone.
    """
    selected = select_lidar_points_in_2d_box(
        points=points,
        calib=calib,
        image_shape=image_shape,
        bbox=bbox,
        expansion=expansion,
    )

    xyz = selected["xyz"]
    depths = selected["depth"]

    row = {
        "n_points_box": int(len(xyz)),
        "has_depth_split": 0,
        "max_depth_gap": 0.0,
        "n_foreground": 0,
        "n_far": 0,
        "foreground_ratio": 0.0,
        "median_depth": float(np.median(depths)) if len(depths) > 0 else np.nan,
    }

    if len(xyz) == 0:
        for name in ["small", "medium", "large"]:
            row[f"{name}_foreground_density"] = 0.0
            row[f"{name}_occupied_voxels"] = 0.0
            row[f"{name}_max_points_per_voxel"] = 0.0

        return row

    split = split_depth_layers_from_proposal(
        depths=depths,
        min_depth_gap=min_depth_gap,
        min_points_per_layer=2,
    )

    if split is None:
        foreground_xyz = np.empty((0, 3), dtype=np.float32)
    else:
        foreground_xyz = xyz[split["near_mask"]]

        row["has_depth_split"] = 1
        row["max_depth_gap"] = split["max_depth_gap"]
        row["n_foreground"] = split["n_near"]
        row["n_far"] = split["n_far"]
        row["foreground_ratio"] = split["n_near"] / max(
            split["n_near"] + split["n_far"],
            1,
        )

    for name, voxel_size in zip(
        ["small", "medium", "large"],
        voxel_sizes,
    ):
        density, occupied_voxels, max_count = compute_voxel_density_for_points(
            foreground_xyz,
            voxel_size=voxel_size,
        )

        row[f"{name}_foreground_density"] = density
        row[f"{name}_occupied_voxels"] = occupied_voxels
        row[f"{name}_max_points_per_voxel"] = max_count

    return row

def fit_robust_params(values, q_low=5, q_high=95):
    values = np.asarray(values, dtype=np.float32)

    low = np.nanpercentile(values, q_low)
    high = np.nanpercentile(values, q_high)

    return {
        "low": float(low),
        "high": float(high),
    }

def apply_robust_params(values, params):
    values = np.asarray(values, dtype=np.float32)

    low = params["low"]
    high = params["high"]

    score = 100.0 * np.clip(
        (values - low) / max(high - low, 1e-6),
        0.0,
        1.0,
    )

    return score

def fit_proposal_or_calibration_params(
    calib_df,
):
    """
    Fit all normalization parameters on calibration data only.
    """
    df = calib_df.copy()

    total_layer_points = df["n_foreground"] + df["n_far"]

    layer_balance = np.where(
        total_layer_points > 0,
        np.minimum(df["n_foreground"], df["n_far"]) / total_layer_points,
        0.0,
    )

    layer_balance_score = 100.0 * np.clip(
        layer_balance / 0.30,
        0.0,
        1.0,
    )

    depth = df["median_depth"].fillna(df["median_depth"].median())
    depth2 = np.square(depth.values)

    balance_gate = layer_balance_score / 100.0

    small_density_corrected = (
        df["small_foreground_density"].values
        * depth2
        * balance_gate
    )

    medium_density_corrected = (
        df["medium_foreground_density"].values
        * depth2
        * balance_gate
    )

    large_density_corrected = (
        df["large_foreground_density"].values
        * depth2
        * balance_gate
    )

    params = {
        "small": fit_robust_params(small_density_corrected),
        "medium": fit_robust_params(medium_density_corrected),
        "large": fit_robust_params(large_density_corrected),
        "gap": fit_robust_params(df["max_depth_gap"].values),
        "median_depth_fill": float(df["median_depth"].median()),
    }

    return params

def score_proposal_density_or_with_params(
    proposal_df,
    params,
    weights=(0.50, 0.30, 0.20),
):
    """
    Score proposal OR using normalization fitted on calibration data.

    No fitting happens here.
    """
    df = proposal_df.copy()

    total_layer_points = df["n_foreground"] + df["n_far"]

    df["layer_balance"] = np.where(
        total_layer_points > 0,
        np.minimum(df["n_foreground"], df["n_far"]) / total_layer_points,
        0.0,
    )

    df["layer_balance_score"] = 100.0 * np.clip(
        df["layer_balance"] / 0.30,
        0.0,
        1.0,
    )

    depth = df["median_depth"].fillna(params["median_depth_fill"])
    depth2 = np.square(depth.values)

    balance_gate = df["layer_balance_score"].values / 100.0

    df["small_density_corrected"] = (
        df["small_foreground_density"].values
        * depth2
        * balance_gate
    )

    df["medium_density_corrected"] = (
        df["medium_foreground_density"].values
        * depth2
        * balance_gate
    )

    df["large_density_corrected"] = (
        df["large_foreground_density"].values
        * depth2
        * balance_gate
    )

    df["Dsmall"] = apply_robust_params(
        df["small_density_corrected"].values,
        params["small"],
    )

    df["Dmedium"] = apply_robust_params(
        df["medium_density_corrected"].values,
        params["medium"],
    )

    df["Dlarge"] = apply_robust_params(
        df["large_density_corrected"].values,
        params["large"],
    )

    df["Dgap"] = apply_robust_params(
        df["max_depth_gap"].values,
        params["gap"],
    )

    df["Dbalance"] = df["layer_balance_score"]

    w_small, w_medium, w_large = weights

    df["density_OR"] = (
        w_small * df["Dsmall"]
        + w_medium * df["Dmedium"]
        + w_large * df["Dlarge"]
    )

    df["proposal_OR"] = (
        0.70 * df["density_OR"]
        + 0.20 * df["Dgap"]
        + 0.10 * df["Dbalance"]
    )

    df.loc[df["has_depth_split"] == 0, "proposal_OR"] = 0.0

    return df
