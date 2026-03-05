import numpy as np
from robot.robot_operation import pick

def pixel_to_robot(u, v, H):
    """Convert pixel coordinates (u, v) to robot coordinates (X, Y) using homography matrix H."""
    # Input: u, v as floats, H as 3x3 homography
    p = np.array([u, v, 1.0], dtype=np.float32).reshape(3, 1)
    pr = H @ p
    pr = pr / pr[2, 0]  # divide by last coordinate to normalize
    X = pr[0, 0]
    Y = pr[1, 0]
    return X, Y

def transform(list, H):
    """Transform a list of pixel coordinates to robot coordinates using the homography matrix."""
    results = []
    for (x, y) in list:
        X_pred, Y_pred = pixel_to_robot(x, y, H)
        print(f"{x},{y} -> {X_pred},{Y_pred}")
        results.append((X_pred, Y_pred))
    return results


def robot_pick(pose_list):
    """Pick objects at the given robot coordinates."""
    pick(pose_list)
