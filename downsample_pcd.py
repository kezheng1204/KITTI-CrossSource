import open3d as o3d
import numpy as np
import glob

from tqdm import tqdm
from pathlib import Path

from config import make_cfg

def read_ply_pointcloud(ply_file_path):
    pcd = o3d.io.read_point_cloud(ply_file_path)
    return pcd


def apply_transform(pcd, T):
    pcd.transform(T)
    return pcd

def save_npy_pointcloud(pcd, path):
    pts = np.asarray(pcd.points)
    pts = pts[:, :3]
    np.save(path, pts)

def read_Tr(calib_path):
    # Read the calibration file
    with open(calib_path, 'r') as f:
        lines = f.readlines()

    Tr_line = lines[4]

    Tr = np.array([float(x) for x in Tr_line.split(' ')[1:13]]).reshape(3, 4)

    # Append the last row
    Tr = np.vstack((Tr, np.array([0, 0, 0, 1])))
    return Tr


cfg = make_cfg()

BASE_DIR = cfg.base_dir
POSE_DIR = cfg.pose_dir
OUT_DIR = cfg.out_dir


for seq in range(11):
    print(f"Processing sequence {seq:02d}")
    seq_out_dir = OUT_DIR / f"{seq:02d}rec"
    seq_out_dir.mkdir(parents=True, exist_ok=True)

    calib_path = BASE_DIR / f"{seq:02d}" / "calib.txt"
    Tr = read_Tr(calib_path)

    with open(POSE_DIR / f"{seq:02d}.txt") as f:
        poses = f.readlines()

    pbar = tqdm(glob.glob(str(BASE_DIR / f"{seq:02d}" / "rec" / "*.ply")))
    for pcd_path in pbar:
        path = Path(pcd_path)
        pbar.set_description(f'Processing {path.name} for {path.name.replace(".ply", ".npy")}')
        
        frame = int(path.name.replace(".ply", ""))
        pose = poses[frame].strip().split()
        pose = np.array(pose).astype(np.float32).reshape(3, 4)
        pose = np.vstack((pose, np.array([0, 0, 0, 1]).reshape(1, 4)))


        pcd = read_ply_pointcloud(pcd_path)
        pcd = pcd.voxel_down_sample(0.3) # voxel size = 0.3
        pcd = apply_transform(pcd, np.linalg.inv(Tr))

        tp = np.linalg.inv(Tr) @ np.linalg.inv(pose) @ Tr
        pcd = apply_transform(pcd, tp)

        save_npy_pointcloud(pcd, seq_out_dir / path.name.replace(".ply", ".npy"))

        