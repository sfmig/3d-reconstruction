
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

### Next steps

Multi-view triangulation:
- https://github.com/strawlab/pymvg/tree/master
- [This pycalib repo](https://github.com/nbhr/pycalib/blob/master/ipynb/ncam_triangulate.ipynb) is educational but probably a good starting point.
   - [Here](https://github.com/nbhr/pycalib/blob/5559e1742f29a5a547c39347825c9acc9c01f0ec/pycalib/calib.py#L307) they have the function to triangulate an array of points given their 2D coordinates in two or more views

Bundle adjustment:
- [What is the difference with triangulation?](https://stackoverflow.com/questions/39745798/whats-the-conceptual-difference-between-bundle-adjustment-and-structure-from-mo)
- This [scipy tutorial](https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
        ) seems useful.
- From the pycalib repo: [this tutorial](https://github.com/nbhr/pycalib/blob/master/ipynb/ncam_ba.ipynb) seems good.



### Some useful resources:
- [OpenCV: Pose computation overview](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)
