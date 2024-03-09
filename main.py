# %%
import cv2
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from utils import CameraData, BeltPoints, plot_rotated_CS_in_WCS

%matplotlib widget
%matplotlib widget

# %%%%%%%%%%%%%
# Camera data
cameras = CameraData()
cameras_specs = cameras.specs
cameras_views = cameras.views

cameras_intrinsics = cameras.intrinsic_matrices

# %%%%%%%%%%%%%%%%
# Belt points
belt_pts = BeltPoints()
belt_coords_CCS = belt_pts.coords_CCS
belt_coords_WCS = belt_pts.coords_WCS

# Check belt points in CCS (1-3)
belt_pts.plot_CCS(cameras)

# Check belt points in WCS 
belt_pts.plot_WCS()



# %%%%%%%%%%%%%%%%%%%%%%%%
# Guess extrinsic matrices
rot_side = np.array([
    [1.,0.,0.,],
    [0.,0.,1.,],
    [0.,-1.,0.,],
])
rot_front = np.array([
    [0.,0.,-1.,],
    [1.,0.,0.,],
    [0.,-1.,0.,],
])
rot_overhead = np.array([
    [1.,0.,0.,],
    [0.,-1.,0.,],
    [0.,0.,-1.,],
])

fig, ax = plot_rotated_CS_in_WCS(rot_side)
ax.set_title('side')

fig, ax = plot_rotated_CS_in_WCS(rot_front)
ax.set_title('front')

fig, ax = plot_rotated_CS_in_WCS(rot_overhead)
ax.set_title('overhead')

rvec_overhead, _ = cv2.Rodrigues(rot_overhead)
rvec_front, _ = cv2.Rodrigues(rot_front)
rvec_side, _ = cv2.Rodrigues(rot_side)
# %%%%%%%%%%%%%%%%%%%%%
# Estimate pose via PnP

# solvePnP
# flags:
# initial guess?
# https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga650ba4d286a96d992f82c3e6dfa525fa

cam = 'overhead'
retval, rvec, tvec = cv2.solvePnP(
    belt_coords_WCS, 
    belt_coords_CCS[cam], 
    cameras_intrinsics[cam], # cameraMatrix,
    np.array([]), # no distorsion
    # rvec=rvec_overhead.copy(),
    # tvec=np.array([[-470.0],[25.0],[70.0]]).copy(),
    # useExtrinsicGuess=True, # cv2.SOLVEPNP_IPPE,
    # flags=cv2.SOLVEPNP_ITERATIVE,
)


rotm, _ = cv2.Rodrigues(rvec)

rotm.T
tvec # vector from O' (origin of rotated frame) to O (WCS)


# # compute full extrinsic
# rotm, _ = cv2.Rodrigues(rvec)
# camera_pose_full = np.vstack(
#     [
#         np.hstack([rotm, tvec]),
#         np.flip(np.eye(1,4))
#     ]
# )
# %%
