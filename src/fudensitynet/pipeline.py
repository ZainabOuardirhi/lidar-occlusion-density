"""Frame/proposal scoring independent of KITTI ground-truth labels."""
import numpy as np
import pandas as pd
from .or_score import compute_raw_proposal_density_or_features,score_proposal_density_or_with_params

def score_frame_proposals(points, calib, image_shape, bboxes, params, expansion=0.05):
    """Score external 2D boxes; no detection is performed here.

    Returns one scored row per bbox in supplied order.
    """
    rows=[]
    for idx,bbox in enumerate(bboxes):
        row=compute_raw_proposal_density_or_features(
            points,calib,image_shape,bbox,expansion=expansion)
        row['proposal_index']=idx
        rows.append(row)
    if not rows: return pd.DataFrame(columns=['proposal_index','proposal_OR'])
    return score_proposal_density_or_with_params(pd.DataFrame(rows),params)

def frame_fusion_required(scored_proposals,threshold):
    """Frame-level rule only: activate fusion if any candidate exceeds OR threshold."""
    return bool((scored_proposals['proposal_OR']>=threshold).any())
