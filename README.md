
## Exploring 3D reconstruction 

For a simple case using `opencv`'s `solvePnP` algorithm.

### To run the notebook locally

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
4. You should be able to run the `compute_camera_extrinsics.ipynb` notebook in the created virtual environment :tada: 

<!-- ### To run the notebook on Github codespaces
1. From the Github main page of the repo, click on Code > Codespaces tab > Open in codespace
2. Open the notebook in the VSCode editor and click on Select kernel (top right)
3. In the pop-up menu, select Install suggested extensions
4. Then click on `Select a Python Environment` > `Create` > `venv` > choose your Python version > Select `requirements.txt` file.
    - This will create a virtual environment called `.venv` -->


### Next steps

Multi-view triangulation:
- https://github.com/strawlab/pymvg/tree/master
- [Gist to triangulate a point seen in n views](https://gist.github.com/davegreenwood/e1d2227d08e24cc4e353d95d0c18c914#file-triangulation-py-L9)
- [This pycalib repo](https://github.com/nbhr/pycalib/blob/master/ipynb/ncam_triangulate.ipynb) is educational but probably a good starting point.
   - [Here](https://github.com/nbhr/pycalib/blob/5559e1742f29a5a547c39347825c9acc9c01f0ec/pycalib/calib.py#L307) they have the function to triangulate an array of points given their 2D coordinates in two or more views

Bundle adjustment:
- [What is the difference with triangulation?](https://stackoverflow.com/questions/39745798/whats-the-conceptual-difference-between-bundle-adjustment-and-structure-from-mo)
- This [scipy tutorial](https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
        ) seems useful.
- From the pycalib repo: [this tutorial](https://github.com/nbhr/pycalib/blob/master/ipynb/ncam_ba.ipynb) seems good.

Triangulate points -- opencv example
- only 2-by-2 views are possible in opencv's Python API 
- see examples [here](https://gist.github.com/davegreenwood/e1d2227d08e24cc4e353d95d0c18c914)
    
### Some useful resources:
- [OpenCV: Pose computation overview](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)
