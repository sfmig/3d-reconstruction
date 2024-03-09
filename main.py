# %%
import cv2
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
# rot_side = np.array([
#     [1.,0.,0.,],
#     [0.,0.,1.,],
#     [0.,-1.,0.,],
# ])
# rot_front = np.array([
#     [0.,0.,-1.,],
#     [1.,0.,0.,],
#     [0.,-1.,0.,],
# ])
# rot_overhead = np.array([
#     [1.,0.,0.,],
#     [0.,-1.,0.,],
#     [0.,0.,-1.,],
# ])

# fig, ax = plot_rotated_CS_in_WCS(rot_side)
# ax.set_title('side')

# fig, ax = plot_rotated_CS_in_WCS(rot_front)
# ax.set_title('front')

# fig, ax = plot_rotated_CS_in_WCS(rot_overhead)
# ax.set_title('overhead')

# rvec_overhead, _ = cv2.Rodrigues(rot_overhead)
# rvec_front, _ = cv2.Rodrigues(rot_front)
# rvec_side, _ = cv2.Rodrigues(rot_side)
# %%%%%%%%%%%%%%%%%%%%%
# Estimate pose via PnP

cameras_extrinsics = cameras.compute_cameras_extrinsics(
    belt_coords_WCS, 
    belt_coords_CCS
)

print('Reprojection errors:')
[(cam, cameras_extrinsics[cam]['repr_err']) for cam in cameras_extrinsics]


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Estimate with initial guess --- REVIEW (front very large)

cameras_extrinsics = cameras.compute_cameras_extrinsics(
    belt_coords_WCS, 
    belt_coords_CCS,
    guess_intrinsics=True
)

print('Reprojection errors:')
[(cam, cameras_extrinsics[cam]['repr_err']) for cam in cameras_extrinsics]




# # %%
# # compute reprojection error
# cam = 'overhead'
# rvec = cameras_extrinsics[cam]['rvec']
# tvec = cameras_extrinsics[cam]['tvec']
# cam_intrinsics = cameras_intrinsics[cam]

# belt_coords_CCS_repr, _ = cv2.projectPoints(
#     belt_coords_WCS,
#     rvec,
#     tvec,
#     cam_intrinsics,
#     np.array([]),  # no distorsion
# )
# belt_coords_CCS_repr = np.squeeze(belt_coords_CCS_repr)
# error = np.mean(
#     np.linalg.norm(
#         belt_coords_CCS_repr - belt_coords_CCS[cam],
#         axis=1
#     )
# )
# error
# %%
