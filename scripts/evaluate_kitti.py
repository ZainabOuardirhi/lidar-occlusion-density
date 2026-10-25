"""Extract KITTI oracle features, fit on frame-disjoint calibration, score test."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from fudensitynet.dataset import build_kitti_oracle_proposal_table
from fudensitynet.evaluation import split_by_sample_id,select_best_threshold,evaluate_final_test_set
from fudensitynet.or_score import fit_proposal_or_calibration_params,score_proposal_density_or_with_params

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--kitti-training',type=Path,required=True,help='Directory with image_2/ velodyne/ calib/ label_2/')
    p.add_argument('--max-frames',type=int,default=500)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output-dir',type=Path,default=Path('runs/or_eval'))
    a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    ids=sorted(x.stem for x in (a.kitti_training/'label_2').glob('*.txt'))[:a.max_frames]
    if not ids: p.error('No KITTI labels found. Check --kitti-training')
    features=build_kitti_oracle_proposal_table(a.kitti_training,ids)
    if features.empty: p.error('No proposals generated')
    features.to_pickle(a.output_dir/'oracle_features.pkl')
    calib,test,*_=split_by_sample_id(features,seed=a.seed)
    params=fit_proposal_or_calibration_params(calib)
    scored_calib=score_proposal_density_or_with_params(calib,params)
    scored_test=score_proposal_density_or_with_params(test,params)
    threshold,_=select_best_threshold(scored_calib)
    if len(set(scored_test.gt_is_occluded))!=2: p.error('Test split has only one binary label; use a larger sample')
    result=evaluate_final_test_set(scored_test,threshold)
    scored_test.to_csv(a.output_dir/'test_proposals.csv',index=False)
    (a.output_dir/'calibration.json').write_text(json.dumps({'params':params,'threshold':threshold,'seed':a.seed},indent=2))
    (a.output_dir/'test_metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
