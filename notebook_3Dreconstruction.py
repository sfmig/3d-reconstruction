# %%
import cv2
# import csv
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# %matplotlib widget
# %matplotlib ipympl
%matplotlib widget
%matplotlib widget

# %%%%%%%%%%%%
# Input data
belt_coords_csv = '/Users/sofia/swc/project_3Dhomography/20240221/belt_corner_coords_80200.csv'
camera_paths = {
    'side': '/Users/sofia/swc/project_3Dhomography/20240221/Side_80200.png',
    'front': '/Users/sofia/swc/project_3Dhomography/20240221/Front_80200.png',
    'overhead': '/Users/sofia/swc/project_3Dhomography/20240221/Overhead_80200.png'
}

# %%%%%%%%%%%%
# Camera specs
camera_specs = {
    'side':{
        'focal_length_mm': 16,
        'height_px': 230,
        'width_px': 1920,
        'pixel_size_x_mm': 4.8*10-3, # 4.8um = 4.8e-3 mm
        'pixel_size_y_mm': 4.8*10-3 # 4.8um = 4.8e-3 mm
    },
    'front':{
        'focal_length_mm': 16,
        'height_px': 230, #ok?
        'width_px': 296, #ok?
        'pixel_size_x_mm': 3.45*10-3, # 4.8um = 4.8e-3 mm
        'pixel_size_y_mm': 3.45*10-3 # 4.8um = 4.8e-3 mm
    },
    'overhead':{
        'focal_length_mm': 12,
        'height_px': 116, #ok?
        'width_px': 992, #ok?
        'pixel_size_x_mm': 3.45*10-3, # 4.8um = 4.8e-3 mm
        'pixel_size_y_mm': 3.45*10-3 # 4.8um = 4.8e-3 mm
    },
}


# Camera views
camera_views = dict()
for cam, path in camera_paths.items():
    camera_views[cam] = plt.imread(path)

# Map point string names to integer names and sort
belt_points_str2int = {
    'StartPlatR': 3,
    'StartPlatL': 0,
    'TransitionR': 2,
    'TransitionL': 1,
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
    ax.set_xlim(0, camera_specs[cam]['width_px'])
    ax.set_ylim(0, camera_specs[cam]['height_px'])
    ax.invert_yaxis()

    # add labels
    ax.set_title(cam)
    ax.set_xlabel('x (px)')
    ax.set_ylabel('y (px)')
    ax.axis('equal')


# fig.show()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# belt points in world coord system
# belt_coords_WCS

# plot

# %%
# Estimate intrinsic matrix of the three cameras


# %%
# Estimate extrinsic matrix 
        
# run pnp for each camera
retval, rvec, tvec = cv.solvePnP(
    objectPoints, 
    imagePoints, 
    cameraMatrix,
)

        


# %%
# Plot camera poses in WCS


# %%%%%%%%%%%%
from mpl_toolkits.mplot3d import axes3d

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Grab some test data.
X, Y, Z = axes3d.get_test_data(0.05)

# Plot a basic wireframe.
ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

plt.show()
# %%
from extrinsic2pyramid.util.camera_pose_visualizer import CameraPoseVisualizer
import numpy as np
%matplotlib widget

# visualiser
visualizer = CameraPoseVisualizer([-50, 50], [-50, 50], [0, 50])

# define camera pose
# argument : extrinsic matrix, color, scaled focal length(z-axis length of frame body of camera
visualizer.extrinsic2pyramid(np.eye(4), 'c', 10)
visualizer.show()
# %%
