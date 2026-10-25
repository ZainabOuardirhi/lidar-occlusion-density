import numpy as np
import pandas as pd
from fudensitynet.or_score import (split_depth_layers_from_proposal,compute_voxel_density_for_points,fit_proposal_or_calibration_params,score_proposal_density_or_with_params)
from fudensitynet.pipeline import frame_fusion_required

def test_gap_split_and_min_points():
    s=split_depth_layers_from_proposal([4,4.5,5,12,12.5,13])
    assert s is not None and s['n_near']==3 and s['n_far']==3
    assert split_depth_layers_from_proposal([3,3.2,3.4,3.8]) is None

def test_voxel_density():
    density,occupied,max_count=compute_voxel_density_for_points(np.array([[0,0,0],[0.01,0,0],[2,2,2]],float),.5)
    assert occupied==2 and max_count==2 and np.isclose(density,12)

def test_no_split_scores_zero():
    df=pd.DataFrame([dict(n_foreground=0,n_far=0,median_depth=10.,has_depth_split=0,max_depth_gap=0.,**{f'{x}_foreground_density':0. for x in ('small','medium','large')})])
    params=fit_proposal_or_calibration_params(df)
    scored=score_proposal_density_or_with_params(df,params)
    assert scored.proposal_OR.iloc[0]==0
    assert not frame_fusion_required(scored,12)

def test_balance_and_weights_exact():
    df=pd.DataFrame([dict(n_foreground=3,n_far=7,median_depth=10.,has_depth_split=1,max_depth_gap=5.,**{f'{x}_foreground_density':1. for x in ('small','medium','large')}),dict(n_foreground=5,n_far=5,median_depth=20.,has_depth_split=1,max_depth_gap=8.,**{f'{x}_foreground_density':2. for x in ('small','medium','large')})])
    params=fit_proposal_or_calibration_params(df)
    scored=score_proposal_density_or_with_params(df,params)
    assert np.allclose(scored.Dbalance,100)
    assert np.allclose(scored.proposal_OR,0.7*scored.density_OR+0.2*scored.Dgap+0.1*scored.Dbalance)
