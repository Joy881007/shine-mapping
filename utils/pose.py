import numpy as np
from numpy.linalg import inv
import csv
from pyquaternion import Quaternion
import torch



def read_calib_file(filename):
    """ 
        read calibration file (with the kitti format)
        returns -> dict calibration matrices as 4*4 numpy arrays
    """
    calib = {}
    calib_file = open(filename)
    key_num = 0

    for line in calib_file:
        # print(line)
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]
        pose = np.zeros((4, 4))

        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        calib[key] = pose

    calib_file.close()
    return calib


def read_poses_file(filename, calibration):
    """ 
        read pose file (with the kitti format)
    """
    pose_file = open(filename)

    poses = []

    Tr = calibration["Tr"]
    Tr_inv = inv(Tr)

    for line in pose_file:
        values = [float(v) for v in line.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        poses.append(
            np.matmul(Tr_inv, np.matmul(pose, Tr))
        )  # lidar pose in world frame

    pose_file.close()
    return poses


def csv_odom_to_transforms(path):

    # odom_tfs = {}
    poses = []
    with open(path, mode="r") as f:
        reader = csv.reader(f)
        # get header and change timestamp label name
        header = next(reader)
        header[0] = "ts"
        # Convert string odometry to numpy transfor matrices
        for row in reader:
            odom = {l: row[i] for i, l in enumerate(header)}
            # Translarion and rotation quaternion as numpy arrays
            trans = np.array([float(odom[l]) for l in ["tx", "ty", "tz"]])
            quat = Quaternion(
                np.array([float(odom[l]) for l in ["qx", "qy", "qz", "qw"]])
            )
            rot = quat.rotation_matrix
            # Build numpy transform matrix
            odom_tf = np.eye(4)
            odom_tf[0:3, 3] = trans
            odom_tf[0:3, 0:3] = rot
            # Add transform to timestamp indexed dictionary
            # odom_tfs[odom["ts"]] = odom_tf
            poses.append(odom_tf)

    return poses

def rt2euler(rt_matrix):
    if isinstance(rt_matrix, torch.Tensor):
        rt_matrix = rt_matrix.clone().cpu().numpy()

    rot_psi = np.arctan2(rt_matrix[2, 1], rt_matrix[2, 2])*180/np.pi
    rot_theta = np.arctan2(-rt_matrix[2, 0], np.sqrt(rt_matrix[2, 1]**2 + rt_matrix[2, 2]**2))*180/np.pi
    rot_phi = np.arctan2(rt_matrix[1,0], rt_matrix[0,0])*180/np.pi
    translation = np.sqrt(rt_matrix[0,3]**2 + rt_matrix[1,3]**2 + rt_matrix[2,3]**2)

    return {"psi": rot_psi, "theta": rot_theta, "phi": rot_phi, "translation": translation}

def calc_pose_error(pose1, pose2):
    '''
    input:
        pose1
            [4, 4]
        pose2
            [4, 4]
    output:
        rot_error
            [float]
        translation_error
            [float]
    '''
    pose1_euler = rt2euler(pose1)
    pose2_euler = rt2euler(pose2)

    psi_error = abs(pose1_euler["psi"] - pose2_euler["psi"]) if abs(pose1_euler["psi"] - pose2_euler["psi"]) < 300 else abs(abs(pose1_euler["psi"] - pose2_euler["psi"]) - 360)
    theta_error = abs(pose1_euler["theta"] - pose2_euler["theta"]) if abs(pose1_euler["theta"] - pose2_euler["theta"]) < 300 else abs(abs(pose1_euler["theta"] - pose2_euler["theta"]) - 360)
    phi_error = abs(pose1_euler["phi"] - pose2_euler["phi"]) if abs(pose1_euler["phi"] - pose2_euler["phi"]) < 300 else abs(abs(pose1_euler["phi"] - pose2_euler["phi"]) - 360)
    rot_error = psi_error + theta_error + phi_error
    translation_error = abs(pose1_euler["translation"] - pose2_euler["translation"])

    return rot_error, translation_error