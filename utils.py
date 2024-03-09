import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2


class CameraData:
    def __init__(self):
        self.view_paths = {
            "side": "/Users/sofia/swc/project_3Dhomography/data/Side_80200.png",
            "front": "/Users/sofia/swc/project_3Dhomography/data/Front_80200.png",
            "overhead": "/Users/sofia/swc/project_3Dhomography/data/Overhead_80200.png",
        }

        self.specs = self.get_cameras_specs()
        self.views = self.get_cameras_views()
        self.intrinsic_matrices = self.get_cameras_intrinsics()
        self.extrinsics_ini_guess = self.get_cameras_extrinsics_guess()

    def get_cameras_specs(self):
        camera_specs = {
            "side": {
                "focal_length_mm": 16,
                "y_size_px": 230,
                "x_size_px": 1920,
                "pixel_size_x_mm": 4.8e-3,  # 4.8um = 4.8e-3 mm
                "pixel_size_y_mm": 4.8e-3,  # 4.8um = 4.8e-3 mm
            },
            "front": {
                "focal_length_mm": 16,
                "y_size_px": 320,  # ok?
                "x_size_px": 296,  # ok?
                "pixel_size_x_mm": 3.45e-3,
                "pixel_size_y_mm": 3.45e-3,
            },
            "overhead": {
                "focal_length_mm": 12,
                "y_size_px": 116,  # ok?
                "x_size_px": 992,  # ok?
                "pixel_size_x_mm": 3.45e-3,
                "pixel_size_y_mm": 3.45e-3,
            },
        }

        return camera_specs

    def get_cameras_views(self):
        camera_views = dict()
        for cam, path in self.view_paths.items():
            camera_views[cam] = plt.imread(path)
        return camera_views

    def get_cameras_intrinsics(self):
        camera_intrinsics = dict()
        for cam in self.specs.keys():
            fx = self.specs[cam]["focal_length_mm"] / self.specs[cam]["pixel_size_x_mm"]
            fy = self.specs[cam]["focal_length_mm"] / self.specs[cam]["pixel_size_y_mm"]
            cx = int(
                self.specs[cam]["x_size_px"] / 2.0
            )  # ---- centre of pixel in centre or corner?
            cy = int(self.specs[cam]["y_size_px"] / 2.0)

            camera_intrinsics[cam] = np.array(
                [
                    [fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0],
                ]
            )
        return camera_intrinsics

    def get_cameras_extrinsics_guess(self):
        # size of volume
        box_length = 470
        box_width = 52
        box_height = 70

        # tvec guess
        # tvec: vector from O' (origin of rotated CS) to WCS
        # assuming cameras centres are approx at center of each side panel
        tvec_guess = dict()
        tvec_guess["side"] = np.array([-box_length / 2, -box_height / 2, box_width]).reshape(-1,1)
        tvec_guess["front"] = np.array([-box_width / 2, -box_height / 2, box_length]).reshape(-1,1)
        tvec_guess["overhead"] = np.array([-box_length / 2, box_width / 2, box_height]).reshape(-1,1)

        # rotm using my definition, aka,
        # in columns, the rotated WCS versors
        # fmt: off
        rot_m_guess = dict()
        rot_m_guess['side'] = np.array(
            [
                [1.0, 0.0, 0.0,],
                [0.0, 0.0, 1.0,],
                [0.0, -1.0, 0.0,],
            ]
        )
        rot_m_guess['front'] = np.array(
            [
                [0.0, 0.0, -1.0,],
                [1.0, 0.0, 0.0,],
                [0.0, -1.0, 0.0,],
            ]
        )
        rot_m_guess['overhead'] = np.array(
            [
                [1.0, 0.0, 0.0,],
                [0.0, -1.0, 0.0,],
                [0.0, 0.0, -1.0,],
            ]
        )
        # fmt: on

        # prepare initial guess for solvePnP
        # rotm.T, rvec, tvec
        # OJO transpose rotm from my definition for opencv!
        cameras_extrinsics_guess = dict()
        for cam in self.specs.keys():
            # Rodrigues vector on rotm with opencv convention
            rodrigues_vec_opencv, _ = cv2.Rodrigues(rot_m_guess[cam].T)

            # save params
            cameras_extrinsics_guess[cam] = {
                "rotm": rot_m_guess[cam].T,
                "rvec": rodrigues_vec_opencv,
                "tvec": tvec_guess[cam],
            }

        return cameras_extrinsics_guess

    def compute_cameras_extrinsics(
        self, belt_coords_WCS, belt_coords_CCS, guess_intrinsics=False
    ):
        camera_extrinsics = dict()
        for cam in self.specs.keys():
            if not guess_intrinsics:
                retval, rvec, tvec = cv2.solvePnP(
                    belt_coords_WCS,
                    belt_coords_CCS[cam],
                    self.intrinsic_matrices[cam],
                    np.array([]),  # no distorsion
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
            else:
                retval, rvec, tvec = cv2.solvePnP(
                    belt_coords_WCS,
                    belt_coords_CCS[cam],
                    self.intrinsic_matrices[cam],
                    np.array([]),  # no distorsion
                    rvec=self.extrinsics_ini_guess[cam]["rvec"].copy(),
                    tvec=self.extrinsics_ini_guess[cam]["tvec"].copy(),
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )

            # OJO!
            # - rotm.T: in columns, versors from WCS rotated
            # - tvec: vector from O' (origin of rotated frame) to O (WCS)

            # compute full extrinsic
            rotm, _ = cv2.Rodrigues(rvec)
            camera_pose_full = np.vstack(
                [np.hstack([rotm, tvec]), np.flip(np.eye(1, 4))]
            )

            # compute reprojection error
            belt_coords_CCS_repr, _ = cv2.projectPoints(
                belt_coords_WCS,
                rvec,
                tvec,
                self.intrinsic_matrices[cam],
                np.array([]),  # no distorsion
            )
            belt_coords_CCS_repr = np.squeeze(belt_coords_CCS_repr)
            error = np.mean(
                np.linalg.norm(belt_coords_CCS_repr - belt_coords_CCS[cam], axis=1)
            )

            # save data
            camera_extrinsics[cam] = {
                "retval": retval,
                "rvec": rvec,
                "tvec": tvec,
                "rotm": rotm,
                "full": camera_pose_full,
                "repr_err": error,
            }

        return camera_extrinsics


class BeltPoints:
    def __init__(self):
        self.coords_csv = (
            "/Users/sofia/swc/project_3Dhomography/data/belt_corner_coords_80200.csv"
        )
        self.points_str2int = {
            "StartPlatR": 0,
            "StartPlatL": 3,
            "TransitionR": 1,
            "TransitionL": 2,
        }
        self.fn_points_str2int = np.vectorize(lambda x: self.points_str2int[x])

        self.coords_CCS = self.get_points_in_CCS()
        self.coords_WCS = self.get_points_in_WCS()

    def get_points_in_WCS(self):
        return np.array(
            [
                [0.0, 0.0, 0.0],
                [470.0, 0.0, 0.0],
                [470.0, 52.0, 0.0],
                [0.0, 52.0, 0.0],
            ]
        )

    def get_points_in_CCS(self):
        # read csv as table
        df = pd.read_csv(self.coords_csv, sep=",")

        # compute idcs to sort points by ID
        points_str_in_input_order = np.array(df.loc[df["coords"] == "x"]["bodyparts"])
        points_IDs_in_input_order = self.fn_points_str2int(points_str_in_input_order)
        sorted_idcs_by_pt_ID = np.argsort(points_IDs_in_input_order)

        sorted_kys = sorted(
            self.points_str2int.keys(), key=lambda ky: self.points_str2int[ky]
        )
        assert all(points_str_in_input_order[sorted_idcs_by_pt_ID] == sorted_kys)

        # loop thru camera views and save points
        list_cameras = list(df.columns[-3:])
        belt_coords_CCS = dict()
        for cam in list_cameras:
            imagePoints = np.array(
                [df.loc[df["coords"] == "x"][cam], df.loc[df["coords"] == "y"][cam]]
            ).T  # imagePoints, NX2

            # sort them by point ID
            belt_coords_CCS[cam] = imagePoints[sorted_idcs_by_pt_ID, :]

        return belt_coords_CCS

    def plot_CCS(self, camera):
        # Check belt points in CCS (1-3)
        fig, axes = plt.subplots(2, 2)

        for cam, ax in zip(camera.specs.keys(), axes.reshape(-1)):
            # add image
            ax.imshow(camera.views[cam])

            # add scatter
            ax.scatter(
                x=self.coords_CCS[cam][:, 0],
                y=self.coords_CCS[cam][:, 1],
                s=50,
                c="r",
                marker="x",
                linewidth=0.5,
                label=range(self.coords_CCS[cam].shape[0]),
            )

            # add image center
            ax.scatter(
                x=camera.views[cam].shape[1] / 2,
                y=camera.views[cam].shape[0] / 2,
                s=50,
                c="b",
                marker="o",
                linewidth=0.5,
                label=range(self.coords_CCS[cam].shape[0]),
            )

            # add text
            for id in range(self.coords_CCS[cam].shape[0]):
                ax.text(
                    x=self.coords_CCS[cam][id, 0],
                    y=self.coords_CCS[cam][id, 1],
                    s=id,
                    c="r",
                )

            # set axes limits,
            ax.set_xlim(0, camera.specs[cam]["x_size_px"])
            ax.set_ylim(0, camera.specs[cam]["y_size_px"])
            ax.invert_yaxis()

            # add labels
            ax.set_title(cam)
            ax.set_xlabel("x (px)")
            ax.set_ylabel("y (px)")
            ax.axis("equal")

    def plot_WCS(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        # add scatter
        ax.scatter(
            self.coords_WCS[:, 0],
            self.coords_WCS[:, 1],
            self.coords_WCS[:, 2],
            s=50,
            c="r",
            marker=".",
            linewidth=0.5,
            alpha=1,
        )

        # add text
        for id in range(self.coords_WCS.shape[0]):
            ax.text(
                self.coords_WCS[id, 0],
                self.coords_WCS[id, 1],
                self.coords_WCS[id, 2],
                s=id,
                c="r",
            )

        for row, col in zip(np.eye(3), ["r", "g", "b"]):
            ax.quiver(
                0,
                0,
                0,
                row[0],
                row[1],
                row[2],
                color=col,
                length=100,
                arrow_length_ratio=0,
                normalize=True,
            )

        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.axis("equal")


def plot_rotated_CS_in_WCS(rot_cam):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # WCS
    for row, col in zip(np.eye(3), ["r", "g", "b"]):
        ax.quiver(
            0,
            0,
            0,
            row[0],
            row[1],
            row[2],
            color=col,
            # length=1,
            arrow_length_ratio=0,
            # normalize=True
        )

    # camera
    for row, col in zip(rot_cam.T, ["r", "g", "b"]):
        ax.quiver(
            0,
            0,
            0,
            row[0],
            row[1],
            row[2],
            color=col,
            # length=1,
            arrow_length_ratio=0,
            # normalize=True,
            linestyle=":",
            linewidth=4,
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    return fig, ax
