import sys
import numpy as np
import cv2
from omni.isaac.core.utils.extensions import enable_extension

# enable_extension("omni.isaac.robot_assembler")

from omni.isaac.core.utils.stage import add_reference_to_stage

# from omni.isaac.robot_assembler import RobotAssembler
from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.sensor import Camera
from omni.isaac.core.prims import RigidPrim, GeometryPrim
from scipy.spatial.transform import Rotation as R
from omni.isaac.core.articulations import Articulation
import matplotlib.pyplot as plt
import inspect
import sys
import tools.transforms as tf
from tools.sample import samplePose
import os
import omni.replicator.core as rep
import omni.isaac.core.utils.prims as prims_utils


class env:
    def __init__(self, root_path, render, physics_dt=1 / 60.0) -> None:
        self.img_count = 0
        self.root_path = root_path
        self.camera_intrinsics = np.array(
            [
                [616.56402588, 0.0, 330.48983765],
                [0.0, 616.59606934, 233.84162903],
                [0.0, 0.0, 1.0],
            ]
        )
        # relative pose
        self.relative_pos_upper = np.array([0.0, 0.0, 0.20])
        self.relative_pos_lower = np.array([0.0, 0.0, 0.20])
        # self.relative_pos_upper = np.array([0.01, 0.01, 0.15])
        # self.relative_pos_lower = np.array([-0.01, -0.01, 0.1])
        self.relative_rot_upper = np.array([0, 0, 0])
        self.relative_rot_lower = np.array([0, 0, 0])
        # self.relative_rot_upper = np.array([np.pi/3, np.pi/3, np.pi/3])
        # self.relative_rot_lower = np.array([-np.pi/3, -np.pi/3, -np.pi/3])

        # in_hand_error_pose
        #TODO test
        self.hand_error_pos_upper = np.array([0.0, 0.0, 0.2])
        self.hand_error_pos_lower = np.array([0.0, 0.0, 0.2])
        # self.hand_error_rot_upper = np.array([np.pi, 0.0, 0.0])
        # self.hand_error_rot_lower = np.array([np.pi, 0.0, 0.0])
        self.hand_error_rot_upper = np.array([np.pi+np.pi / 12, np.pi / 12, np.pi])
        self.hand_error_rot_lower = np.array([np.pi-np.pi / 12, -np.pi / 12, -np.pi])

        # self.hand_error_pos_upper = np.array([0.02, 0.0, 0.14])
        # self.hand_error_pos_lower = np.array([-0.02, 0.0, 0.12])
        # self.hand_error_rot_upper = np.array([0.0, 0.0, -3.14])
        # self.hand_error_rot_lower = np.array([0.0, 0.3, 3.14])

        self.render = render

        self.world = World(physics_dt=physics_dt, stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()

        hole_asset_path = os.path.join(root_path, "assets/pocketbook_pro_602_obj/pad.usd")
        peg_asset_path = os.path.join(root_path, "assets/iphone/iphone.usd")

        print("peg_asset_path: ", peg_asset_path)
        print("hole_asset_path: ", hole_asset_path)
        self.camera_to_end_effector_pos = np.array([0.0, 0.1, 0.0])
        self.camera_to_end_effector_ori = R.from_euler(
            "xzx", [np.pi, np.pi/2, np.pi/2 + np.pi/15]
        ).as_matrix()
        self.camera_to_end_effector_ori = tf.rot_matrix_to_quat(
            self.camera_to_end_effector_ori
        )
        self.camera_to_end_effector_trans_matrix = tf.calc_trans_matrix(
            self.camera_to_end_effector_pos, self.camera_to_end_effector_ori
        )
        default_light = prims_utils.get_prim_at_path(
            prim_path="/World/defaultGroundPlane/SphereLight"
        )
        default_light.GetAttribute("radius").Set(1.0)
        default_light.GetAttribute("intensity").Set(10000)
        # prims_utils.create_prim(
        #     "/World/light",
        #     "DemoLight",
        #     attributes={"inputs:intensity": 1000},
        # )
        prims_utils.create_prim(
            prim_path="/World/peg",
            prim_type="Xform",
            # usd_path=peg_asset_path,
            # position=np.array([0.0, 0.0, 0.1]),
            # orientation=np.array([1, 0, 0, 0]),
            semantic_label="peg",
        )
        prims_utils.create_prim(
            prim_path="/World/hole",
            prim_type="Xform",
            # usd_path=hole_asset_path,
            # position=np.array([0.0, 0.0, 0.00877]),
            # orientation=tf.euler_angles_to_quat(
            # np.array([-np.pi / 2, 0.0, 0.0]), extrinsic=False
            # ),
            semantic_label="hole",
        )
        
        self.peg = self.add_object(
            usd_path=peg_asset_path,
            prim_path="/World/peg",
            name="peg",
            fixed=True,
            disable_stablization=False,
            mass=0.1,
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([1, 0, 0, 0]),
        )
        # self.camera = rep.create.camera()
        # self.render_product = rep.create.render_product(self.camera, (640, 480))
        # self.instance_seg = rep.AnnotatorRegistry.get_annotator("instance_segmentation")
        # self.instance_seg.attach(self.render_product)
        
        self.camera = Camera(
            prim_path="/World/camera",
            position=np.array([0.0, 0.0, 0.0]),
            frequency=30,
            resolution=(640, 480),
            orientation=tf.euler_angles_to_quat(np.array([0, 0, 0])),
        )
        self.hole = self.add_object(
            usd_path=hole_asset_path,
            prim_path="/World/hole",
            name="hole",
            fixed=False,
            collision=True,
            approx="convexHull",
            disable_stablization=False,
            mass=0.1,
            position=np.array([0.0, 0.0, 0.0]),
            # position=np.array([0.0, 0.0, 0.2]),
            orientation=tf.euler_angles_to_quat(
                np.array([0.0, 0.0, 0.0]), extrinsic=False
            ),
        )

        # self.peg = self.world.scene.add(peg)
        # self.hole = self.world.scene.add(hole)
        self.reset()

    def setCameraParam(self):
        width, height = 640, 480
        pixel_size = 3e-3
        ((fx, _, cx), (_, fy, cy), (_, _, _)) = self.camera_intrinsics
        horizontal_aperture = pixel_size * width
        vertical_aperture = pixel_size * height
        focal_length_x = fx * pixel_size
        focal_length_y = fy * pixel_size
        focal_length = (focal_length_x + focal_length_y) / 2
        self.camera.set_resolution((width, height))
        self.camera.set_focal_length(focal_length / 10.0)
        self.camera.set_horizontal_aperture(horizontal_aperture / 10.0)
        self.camera.set_vertical_aperture(vertical_aperture / 10.0)
        self.camera.set_clipping_range(0.01, 10)
        self.camera.add_semantic_segmentation_to_frame()

    def reset(self):
        self.count = 0

        hand_error_pos, hand_error_ori, hand_error_T = samplePose(
            self.hand_error_pos_upper,
            self.hand_error_pos_lower,
            self.hand_error_rot_upper,
            self.hand_error_rot_lower,
        )
        # self.robot = Articulation("/World/gripper")
        self.world.reset()
        # self.peg.disable_gravity()
        self.camera.initialize()
        self.setCameraParam()

        # relative_T = np.eye(4)
        hole_pos, hole_q = self.hole.get_world_pose()
        hole_T = tf.calc_trans_matrix(hole_pos, hole_q)
        default_T = tf.calc_trans_matrix(
            np.array([0.0, 0.0, 0.0]),
            tf.euler_angles_to_quat(np.array([0.0, 0.0, 0.0]), extrinsic=False),
        )
        _, _, self.relative_T = samplePose(
            self.relative_pos_upper,
            self.relative_pos_lower,
            self.relative_rot_upper,
            self.relative_rot_lower,
        )
        self.relative_pos = self.relative_T[:3, 3]
        self.relative_q = tf.rot_matrix_to_quat(self.relative_T[:3, :3])
        self.peg_T = hole_T @ default_T @ self.relative_T
        self.peg_pos = self.peg_T[:3, 3]
        self.peg_q = tf.rot_matrix_to_quat(self.peg_T[:3, :3])
        gripper_T = self.peg_T @ hand_error_T
        camera_T = gripper_T @ self.camera_to_end_effector_trans_matrix
        print(camera_T)
        camera_q = tf.rot_matrix_to_quat(camera_T[:3, :3])
        camera_pos = camera_T[:3, 3]
        self.peg.set_world_pose(self.peg_pos, self.peg_q)
        self.set_camera_pose()

    def set_camera_pose(self):
        hand_error_pos, hand_error_ori, hand_error_T = samplePose(
            self.hand_error_pos_upper,
            self.hand_error_pos_lower,
            self.hand_error_rot_upper,
            self.hand_error_rot_lower,
        )
        gripper_T = self.peg_T @ hand_error_T
        camera_T = gripper_T @ self.camera_to_end_effector_trans_matrix
        camera_q = tf.rot_matrix_to_quat(camera_T[:3, :3])
        camera_pos = camera_T[:3, 3]
        print(camera_pos)
        self.camera.set_world_pose(camera_pos, camera_q)
        
    def run(self):
        
        self.world.step(render=self.render)
        if self.world.is_playing():
            if self.world.current_time_step_index == 0:
                self.reset()
            else:
                if self.world.current_time_step_index > 30:
                    self.count += 1
                    # pass
                    self.set_camera_pose()
                    img = self.camera.get_rgba()[:, :, :3]
                    
                    segment_data = self.camera._custom_annotators[
                        "semantic_segmentation"
                    ].get_data()
                    segment_id = {
                        v["class"]: int(k)
                        for k, v in segment_data["info"]["idToLabels"].items()
                    }
                    seg_img = np.zeros_like(img)
                    if "peg" in segment_id.keys():
                        peg_seg_idx = segment_data["data"] == segment_id["peg"]
                        seg_img[peg_seg_idx] = img[peg_seg_idx]
                        if self.count > 10:
                            cv2.imwrite(
                            os.path.join(
                                "/cyx/data/vs/debug/img",
                                "img-{}.png".format(self.img_count),
                            ),
                            img,
                            )
                            cv2.imwrite(
                                os.path.join(
                                    "/cyx/data/vs/debug/seg",
                                    "img-{}.png".format(self.img_count),
                                ),
                                seg_img,
                            )
                            self.img_count += 1
                        
                    if self.img_count == 10:
                        exit()


    def add_object(
        self,
        usd_path: str,
        prim_path: str,
        name: str,
        fixed: bool = False,
        collision: bool = False,
        approx: str = "none",
        **kwargs
    ):
        add_reference_to_stage(usd_path, prim_path)

        geo_req_kwargs = inspect.signature(GeometryPrim).parameters
        geo_get_kwargs = {k: v for k, v in kwargs.items() if k in geo_req_kwargs}
        print("RigidPrim kwargs:")
        print(geo_get_kwargs)
        obj = GeometryPrim(
            prim_path=prim_path, name=name, collision=collision, **geo_get_kwargs
        )

        if collision:
            obj.set_collision_approximation(approx)

        if not fixed:
            rigid_req_kwargs = inspect.signature(RigidPrim).parameters
            rigid_get_kwargs = {
                k: v for k, v in kwargs.items() if k in rigid_req_kwargs
            }
            print("RigidPrim kwargs:")
            print(rigid_get_kwargs)
            RigidPrim.__init__(
                obj, prim_path=obj.prim_path, name=obj.name, **rigid_get_kwargs
            )
        return obj
