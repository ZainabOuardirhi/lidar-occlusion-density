"""Phase 1 research reference implementation extracted from implementation(1).ipynb.
Original KITTI-oracle proposal OR equations retained.
"""
import numpy as np

def load_velodyne_bin(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)

def read_kitti_calib(path):
    calib = {}
    with open(path, "r") as f:
        for line in f.readlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            calib[key] = np.array([float(x) for x in value.split()], dtype=np.float32)

    P2 = calib["P2"].reshape(3, 4)
    R0_rect = np.eye(4, dtype=np.float32)
    R0_rect[:3, :3] = calib["R0_rect"].reshape(3, 3)

    Tr_velo_to_cam = np.eye(4, dtype=np.float32)
    Tr_velo_to_cam[:3, :] = calib["Tr_velo_to_cam"].reshape(3, 4)

    return {
        "P2": P2,
        "R0_rect": R0_rect,
        "Tr_velo_to_cam": Tr_velo_to_cam,
    }

def read_kitti_labels(path):
    objects = []
    with open(path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 15:
                continue

            obj = {
                "class": parts[0],
                "truncation": float(parts[1]),
                "occlusion": int(parts[2]),
                "alpha": float(parts[3]),
                "bbox": [float(x) for x in parts[4:8]],  # x1,y1,x2,y2
                "dimensions": [float(x) for x in parts[8:11]],  # h,w,l
                "location": [float(x) for x in parts[11:14]],    # x,y,z in camera coord
                "rotation_y": float(parts[14]),
            }
            objects.append(obj)

    return objects

def project_lidar_to_image(points_xyz, calib):
    n = points_xyz.shape[0]
    pts_hom = np.hstack([points_xyz, np.ones((n, 1), dtype=np.float32)])

    pts_cam = (calib["R0_rect"] @ calib["Tr_velo_to_cam"] @ pts_hom.T).T
    depth = pts_cam[:, 2]

    pts_img_hom = (calib["P2"] @ pts_cam.T).T
    uv = pts_img_hom[:, :2] / np.maximum(pts_img_hom[:, 2:3], 1e-6)

    valid = depth > 0
    return uv, depth, valid
