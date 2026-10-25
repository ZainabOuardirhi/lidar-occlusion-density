# Experimental results

## KITTI oracle-proposal evaluation

The recorded experiment evaluates proposal-level OR against KITTI object occlusion annotations. KITTI ground-truth bounding boxes define the regions during this evaluation; they are not detector predictions. Labels 0 and 1/2 form the visible and occluded groups.

Five frame-level calibration/test splits (seeds 42–46) were reported in the research notebook and Antwerp presentation. The aggregate values are listed in `multisplit_metrics.csv`.

The score uses the calibration-fitted normalization in `src/fudensitynet/or_score.py`, and the reported calibrated threshold was 12 across the five splits. These figures are historical experimental values, not a new run of the packaged release.

The multiscale and depth figures are in `../figures/`. The RGB proposals used to illustrate difficult visibility conditions are selected using KITTI annotations. They are visual examples, not detector output.

### Figures

| Topic | File |
| --- | --- |
| Whole-scene density | [`002050_slide4_left_whole_scene_density.png`](../figures/002050_slide4_left_whole_scene_density.png) |
| Occluded object proposals | [`002050_slide4_right_occluded_proposals.png`](../figures/002050_slide4_right_occluded_proposals.png) |
| Object proposal and LiDAR projection | [`002050_slide5_combined_proposal_level_analysis_v2.png`](../figures/002050_slide5_combined_proposal_level_analysis_v2.png) |
| Depth-layer split | [`002050_slide6_sorted_depth_gap_annotated.png`](../figures/002050_slide6_sorted_depth_gap_annotated.png) |
| Projected near and far LiDAR | [`002050_slide6_03_projected_near_far_layers.png`](../figures/002050_slide6_03_projected_near_far_layers.png) |
| Multiscale visualization | [`002050_slide7_multiscale_voxel_density.html`](../figures/002050_slide7_multiscale_voxel_density.html) |
| Distance-aware proposal illustration | [`002050_distance_aware_OR_proposals.png`](../figures/002050_distance_aware_OR_proposals.png) |

### Notes on interpretation

The OR score is a geometric quantity, not a probability. The distance-aware illustrations show a proposed priority adjustment but do not establish a detection improvement. An earlier illustrative OR boxplot in the presentation comes from a different scoring run and is not included as a matched visualization of the five-split metrics. When reproducing experiments, regenerate distribution plots from the scored evaluation dataframe.
