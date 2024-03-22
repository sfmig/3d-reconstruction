
## Exploring 3D reconstruction 

For a simple case using `opencv`'s `solvePnP` algorithm.

### To run the notebook `compute_camera_extrinsics.ipynb`

1. Create a Python virtual environment. 
    For example to use `conda` to create an environment called `3d-reconstruction`:
    ```
    conda create -n 3d-reconstruction python=3.10
    ```
2. Activate the virtual environment. 
   With conda:
    ```
    conda activate 3d-reconstruction
    ```
3. Install the dependencies specified in the `requirements.txt` file with:
    ```
    pip install -r requirements.txt 
    ```
4. You should be able to run the notebook in the created virtual environment :tada: 

### Some useful resources:
- [OpenCV: Pose computation overview](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)
