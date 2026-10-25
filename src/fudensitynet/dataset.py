"""KITTI oracle-proposal validation (ground-truth boxes not used at inference)."""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from .kitti import load_velodyne_bin,read_kitti_calib,read_kitti_labels
from .or_score import compute_raw_proposal_density_or_features

def build_kitti_oracle_proposal_table(root,sample_ids,allowed_classes=('Car','Pedestrian','Cyclist'),expansion=0.05):
    root=Path(root); rows=[]
    for sample_id in sample_ids:
        sid=str(sample_id).zfill(6)
        image_path=root/'image_2'/f'{sid}.png'
        if not image_path.is_file():continue
        with Image.open(image_path) as im: image_shape=(im.height,im.width,3)
        points=load_velodyne_bin(str(root/'velodyne'/f'{sid}.bin'))
        calib=read_kitti_calib(str(root/'calib'/f'{sid}.txt'))
        objects=read_kitti_labels(str(root/'label_2'/f'{sid}.txt'))
        for object_index,obj in enumerate(objects):
            if obj['class'] not in allowed_classes or obj['occlusion']==3:continue
            row={'sample_id':sid,'object_index':object_index,'class':obj['class'],
                'bbox':obj['bbox'],'gt_occlusion':int(obj['occlusion']),
                'gt_is_occluded':int(obj['occlusion'] in [1,2])}
            row.update(compute_raw_proposal_density_or_features(points,calib,image_shape,obj['bbox'],expansion=expansion))
            rows.append(row)
    return pd.DataFrame(rows)
