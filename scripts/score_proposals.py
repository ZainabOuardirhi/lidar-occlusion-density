"""Example inference-time scoring of externally supplied proposal boxes."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
from fudensitynet.kitti import read_kitti_calib,load_velodyne_bin
from fudensitynet.pipeline import score_frame_proposals,frame_fusion_required

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--velodyne',type=Path,required=True)
    p.add_argument('--calib',type=Path,required=True)
    p.add_argument('--boxes',type=Path,required=True,help='JSON array of [x1,y1,x2,y2] candidate boxes')
    p.add_argument('--calibration',type=Path,required=True,help='Saved calibration.json from evaluate_kitti.py')
    a=p.parse_args(); config=json.loads(a.calibration.read_text())
    boxes=json.loads(a.boxes.read_text())
    with Image.open(a.image) as im: shape=(im.height,im.width,3)
    scored=score_frame_proposals(load_velodyne_bin(str(a.velodyne)),read_kitti_calib(str(a.calib)),shape,boxes,config['params'])
    print(scored[['proposal_index','proposal_OR']].to_json(orient='records',indent=2))
    print('fusion_required:',frame_fusion_required(scored,config['threshold']))
    print('NOTE: this is a routing recommendation, not multimodal inference.')

if __name__=='__main__':main()
