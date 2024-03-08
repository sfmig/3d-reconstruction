# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Imports
import cv2
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

%matplotlib widget
%matplotlib widget

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data
belt_coords_csv = '/Users/sofia/swc/project_3Dhomography/data/belt_corner_coords_80200.csv'
camera_paths = {
    'side': '/Users/sofia/swc/project_3Dhomography/data/Side_80200.png',
    'front': '/Users/sofia/swc/project_3Dhomography/data/Front_80200.png',
    'overhead': '/Users/sofia/swc/project_3Dhomography/data/Overhead_80200.png'
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Camera specs
camera_specs = {
    'side':{
        'focal_length_mm': 16,
        'y_size_px': 230,
        'x_size_px': 1920,
        'pixel_size_x_mm': 4.8*10-3, # 4.8um = 4.8e-3 mm
        'pixel_size_y_mm': 4.8*10-3 # 4.8um = 4.8e-3 mm
    },
    'front':{
        'focal_length_mm': 16,
        'y_size_px': 320, #ok?
        'x_size_px': 296, #ok?
        'pixel_size_x_mm': 3.45*10-3, 
        'pixel_size_y_mm': 3.45*10-3 
    },
    'overhead':{
        'focal_length_mm': 12,
        'y_size_px': 116, #ok?
        'x_size_px': 992, #ok?
        'pixel_size_x_mm': 3.45*10-3, 
        'pixel_size_y_mm': 3.45*10-3
    },
}


# Camera views
camera_views = dict()
for cam, path in camera_paths.items():
    camera_views[cam] = plt.imread(path)

# Map point string names to integer names and sort
belt_points_str2int = {
    'StartPlatR': 0,
    'StartPlatL': 3,
    'TransitionR': 1,
    'TransitionL': 2,
}

fn_belt_points_str2int = np.vectorize(
    lambda x: belt_points_str2int[x]
)

sorted_kys = sorted(
        belt_points_str2int.keys(), 
        key=lambda ky: belt_points_str2int[ky]
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Belt points in Camera coord systems (CCS)

# read csv as table
df = pd.read_csv(belt_coords_csv, sep=',') #, header=None)

# compute idcs to sort points by ID
points_str_in_input_order = np.array(df.loc[df['coords']=='x']['bodyparts'])
points_IDs_in_input_order = fn_belt_points_str2int(points_str_in_input_order)
sorted_idcs_by_pt_ID = np.argsort(points_IDs_in_input_order)

assert all(points_str_in_input_order[sorted_idcs_by_pt_ID] == sorted_kys)

# loop thru camera views and save points
list_cameras = camera_specs.keys()
belt_coords_CCS = dict()
for cam in list_cameras:
    imagePoints = np.array(
        [
            df.loc[df['coords']=='x'][cam],
            df.loc[df['coords']=='y'][cam]
        ] 
    ).T # imagePoints, NX2

    # sort them by point ID
    belt_coords_CCS[cam] =  imagePoints[sorted_idcs_by_pt_ID, :]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Check belt points in CCS (1-3)
fig, axes = plt.subplots(2,2)

for cam, ax in zip(list_cameras, axes.reshape(-1)):
    # add image
    ax.imshow(camera_views[cam])

    # add scatter
    ax.scatter(
        x=belt_coords_CCS[cam][:,0],
        y=belt_coords_CCS[cam][:,1],
        s=50,
        c='r',
        marker='x',
        linewidth=.5,
        label=range(belt_coords_CCS[cam].shape[0])
    )

    # add text
    for id in range(belt_coords_CCS[cam].shape[0]):
        ax.text(
            x=belt_coords_CCS[cam][id,0],
            y=belt_coords_CCS[cam][id,1],
            s=id,
            c='r'
        )

    # set axes limits,
    ax.set_xlim(0, camera_specs[cam]['x_size_px'])
    ax.set_ylim(0, camera_specs[cam]['y_size_px'])
    ax.invert_yaxis()

    # add labels
    ax.set_title(cam)
    ax.set_xlabel('x (px)')
    ax.set_ylabel('y (px)')
    ax.axis('equal')


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Belt points in world coord system
belt_coords_WCS = np.array([
    [0.0, 0.0, 0.0],
    [470.0, 0.0, 0.0],
    [470.0, 52.0, 0.0],
    [0.0, 52.0, 0.0],
])

# plot 3D
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# add scatter
ax.scatter(
    belt_coords_WCS[:,0],
    belt_coords_WCS[:,1],
    belt_coords_WCS[:,2],
    s=50,
    c='r',
    marker='.',
    linewidth=.5,
    alpha=1,
)

# add text
for id in range(belt_coords_WCS.shape[0]):
    ax.text(
        belt_coords_WCS[id,0],
        belt_coords_WCS[id,1],
        belt_coords_WCS[id,2],
        s=id,
        c='r'
)

for row, col in zip(np.eye(3), ['r','g','b']):
    ax.quiver(
        0, 0, 0, 
        row[0], row[1], row[2], 
        color=col,
        length=100, 
        arrow_length_ratio=0,
        normalize=True
    )

ax.set_xlabel('x (mm)')
ax.set_ylabel('y (mm)')
ax.axis('equal')
# %% Estimate extrinsic matrix?

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Estimate intrinsic matrix of the three cameras

camera_intrinsics = dict()
for cam in camera_specs.keys():

    fx = camera_specs[cam]['focal_length_mm']/camera_specs[cam]['pixel_size_x_mm']
    fy = camera_specs[cam]['focal_length_mm']/camera_specs[cam]['pixel_size_y_mm']
    cx = int(camera_specs[cam]['x_size_px']/2.0) #---- centre of pixel in centre or corner?
    cy = int(camera_specs[cam]['y_size_px']/2.0)

    camera_intrinsics[cam] = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ]
    )

camera_intrinsics
# %%%%%%%%%%
# Guess extrinsics    
# from extrinsic2pyramid.util.camera_pose_visualizer import CameraPoseVisualizer
# import numpy as np
# %matplotlib widget


# # side
# side_rot = np.array(
#     [
#         [1,0,0],
#         [0,1,0],
#         [0,0,1],
#     ],
#     dtype=float
# )
# side_trans = np.array([0,0,0,1]).reshape(-1,1)
# side_full = np.hstack(
#     [
#         np.vstack([side_rot, [0,0,0]]),
#         side_trans
#     ]
# )

# # visualiser
# visualizer = CameraPoseVisualizer([-50, 50], [-50, 50], [0, 50])

# visualizer.extrinsic2pyramid(side_full, 'c', 50)
# visualizer.show()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Estimate extrinsic matrices
        
# run pnp for each camera
camera_extrinsics = dict()
for cam in camera_specs.keys(): #['overhead']: #

    # solvePnP
    # flags:
    # initial guess?
    # https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga650ba4d286a96d992f82c3e6dfa525fa
    retval, rvec, tvec = cv2.solvePnP(
        belt_coords_WCS, 
        belt_coords_CCS[cam], 
        camera_intrinsics[cam], # cameraMatrix,
        np.array([]), # no distorsion
        rvec=np.array([[0.0],[0.0],[-np.pi]]),
        tvec=np.array([[-235.0],[-50.0],[25.0]]),
        useExtrinsicGuess=True, # cv2.SOLVEPNP_IPPE,
        flags=cv2.SOLVEPNP_IPPE,
        # cv2.SOLVEPNP_IPPE,
        # cv2.SOLVEPNP_P3P
    )

    # compute full extrinsic
    rotm, _ = cv2.Rodrigues(rvec)
    camera_pose_full = np.vstack(
        [
            np.hstack([rotm, tvec]),
            np.flip(np.eye(1,4))
        ]
    )

    # reprojection error
    belt_coords_CCS_repr, _ = cv2.projectPoints(
        belt_coords_WCS, 
        rvec, 
        tvec, 
        camera_intrinsics[cam], 
        np.array([]), # no distorsion
    )
    belt_coords_CCS_repr = np.squeeze(belt_coords_CCS_repr)
    error = np.sum(
        np.linalg.norm(
            belt_coords_CCS[cam] - belt_coords_CCS_repr, 
            axis = 1
        )
    )/belt_coords_CCS[cam].shape[0]
    # error = cv2.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        
    # save data
    camera_extrinsics[cam] = {
        'retval': retval,
        'rvec': rvec,
        'tvec': tvec,
        'rotm': rotm,
        'full': camera_pose_full,
        'repr_err': error
    }

camera_extrinsics['overhead']['rotm']
# camera_extrinsics['overhead']['tvec']

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot camera poses in WCS

# plot 3D


for cam in camera_extrinsics.keys():
    tvec = camera_extrinsics[cam]['tvec']

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # add wcs
    # ax.quiver([0,0,0], [0,0,0], [0,0,0], [5,0,0], [0,5,0], [0,0,5], length=0.1, normalize=False)
    for row, col in zip(np.eye(3), ['r','g','b']):
        ax.quiver(
            0, 0, 0, 
            row[0], row[1], row[2], 
            color=col,
            length=100, 
            arrow_length_ratio=0,
            normalize=True
        )


    # add belt points
    ax.scatter(
        belt_coords_WCS[:,0],
        belt_coords_WCS[:,1],
        belt_coords_WCS[:,2],
        s=50,
        c='r',
        marker='.',
        linewidth=.5,
        alpha=1,
    )

    # add text
    for id in range(belt_coords_WCS.shape[0]):
        ax.text(
            belt_coords_WCS[id,0],
            belt_coords_WCS[id,1],
            belt_coords_WCS[id,2],
            s=id,
            c='r'
    )

    # add camera translations
    # for cam in ['side']: #camera_extrinsics.keys():
    #     tvec = camera_extrinsics[cam]['tvec']
    ax.scatter(
        tvec[0,:],
        tvec[1,:],
        tvec[2,:],
        s=500,
        c='b',
        marker='.',
        linewidth=.5,
        alpha=1,
    )


    # add labels
    ax.set_title(cam)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')
    ax.axis('equal')


# %%%%%%%%%%
from extrinsic2pyramid.util.camera_pose_visualizer import CameraPoseVisualizer
visualizer = CameraPoseVisualizer([-500, 500], [-500, 500], [0, 500])

# define camera pose
# argument : extrinsic matrix, color, scaled focal length(z-axis length of frame body of camera
visualizer.extrinsic2pyramid(camera_extrinsics['overhead']['full'], 'c', 100)
visualizer.show()



# %%%%%%%%%%%%
from extrinsic2pyramid.util.camera_pose_visualizer import CameraPoseVisualizer

# visualiser
visualizer = CameraPoseVisualizer([-50, 50], [-50, 50], [0, 50])

# define camera pose
# argument : extrinsic matrix, color, scaled focal length(z-axis length of frame body of camera
rot_m = np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0, 1]
])
rot_mx = np.array([
    [1,0,0,0],
    [0,0,-1,0],
    [0,1,0,0],
    [0,0,0, 1]
])
visualizer.extrinsic2pyramid(rot_mx, 'c', 10)


visualizer.show()
# %%%%%%%

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# rot 90deg around x axis
# in cols: rotated versors
theta = -30*np.pi/180
rot_mx = np.array([
    [1,0,0,],
    [0,np.cos(theta),-np.sin(theta),],
    [0,np.sin(theta),np.cos(theta),],
])

for row, col in zip(np.eye(3), ['r','g','b']):
    ax.quiver(
        0, 0, 0, 
        row[0], row[1], row[2], 
        color=col,
        # length=1, 
        arrow_length_ratio=0,
        # normalize=True
    )
for row, col in zip(rot_mx.T, ['m','y','c']):
    ax.quiver(
        0, 0, 0, 
        row[0], row[1], row[2], 
        color=col,
        # length=1, 
        arrow_length_ratio=0,
        # normalize=True,
        linestyle=':'
    )
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
# ax.axis('equal')
# %%
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# rot 90deg around x axis
# in cols: rotated versors
theta = -30*np.pi/180
rot_front = np.array([
    [0,0,-1,],
    [1,0,0,],
    [0,-1,0,],
])

for row, col in zip(np.eye(3), ['r','g','b']):
    ax.quiver(
        0, 0, 0, 
        row[0], row[1], row[2], 
        color=col,
        # length=1, 
        arrow_length_ratio=0,
        # normalize=True
    )
for row, col in zip(rot_front.T, ['m','y','c']):
    ax.quiver(
        0, 0, 0, 
        row[0], row[1], row[2], 
        color=col,
        # length=1, 
        arrow_length_ratio=0,
        # normalize=True,
        # linestyle=':'
    )
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
# %%
