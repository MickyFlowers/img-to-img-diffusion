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
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import inspect
import sys
import tools.transforms as tf
from tools.sample import samplePose
import torch
import torchvision
import os
import omni.replicator.core as rep
import omni.isaac.core.utils.prims as prims_utils
import argparse
import os
import warnings
import torch.multiprocessing as mp
# import copy
from core.logger import VisualWriter, InfoLogger
import core.praser as Praser
import core.util as Util
from data import define_dataloader
from models import create_model, define_network, define_loss, define_metric
from PIL import Image
from tools.perception import Frame, PinholeCamera
from tools.ibvs import IBVS
import tools.action as action
def main_worker(gpu, ngpus_per_node, opt):
    """threads running on each GPU"""
    if "local_rank" not in opt:
        opt["local_rank"] = opt["global_rank"] = gpu
        
    if opt["distributed"]:
        torch.cuda.set_device(int(opt["local_rank"]))
        print("using GPU {} for training".format(int(opt["local_rank"])))
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=opt["init_method"],
            world_size=opt["world_size"],
            rank=opt["global_rank"],
            group_name="mtorch",
        )
    """set seed and and cuDNN environment """
    torch.backends.cudnn.enabled = True
    warnings.warn(
        "You have chosen to use cudnn for accleration. torch.backends.cudnn.enabled=True"
    )
    Util.set_seed(opt["seed"])

    """ set logger """
    phase_logger = InfoLogger(opt)
    phase_writer = VisualWriter(opt, phase_logger)
    phase_logger.info(
        "Create the log file in directory {}.\n".format(opt["path"]["experiments_root"])
    )

    """set networks and dataset"""
    phase_loader, val_loader = define_dataloader(
        phase_logger, opt
    )  # val_loader is None if phase is test.
    networks = [
        define_network(phase_logger, opt, item_opt)
        for item_opt in opt["model"]["which_networks"]
    ]

    """ set metrics, loss, optimizer and  schedulers """
    metrics = [
        define_metric(phase_logger, item_opt)
        for item_opt in opt["model"]["which_metrics"]
    ]
    losses = [
        define_loss(phase_logger, item_opt) for item_opt in opt["model"]["which_losses"]
    ]

    model = create_model(
        opt=opt,
        networks=networks,
        phase_loader=phase_loader,
        val_loader=val_loader,
        losses=losses,
        metrics=metrics,
        logger=phase_logger,
        writer=phase_writer,
    )
    phase_logger.info("Begin model {}.".format(opt["phase"]))    
    
    return model

class env:
    def __init__(self, root_path, render, physics_dt=1 / 60.0) -> None:
        self.count = 0 
        self.eva_count = 0
        self.success_count = 0
        self.total_pos_error = 0
        self.total_rot_error = 0
        self.root_path = root_path
        self.camera_intrinsics = np.array(
            [
                [616.56402588, 0.0, 330.48983765],
                [0.0, 616.59606934, 233.84162903],
                [0.0, 0.0, 1.0],
            ]
        )
        self.usd2opencv = np.array([[1.0,  0.0,  0.0, 0.0],
                                    [0.0, -1.0,  0.0, 0.0],
                                    [0.0,  0.0, -1.0, 0.0],
                                    [0.0,  0.0,  0.0, 1.0]])
        self.opencv2usd = np.linalg.inv(self.usd2opencv)
        # relative pose
        self.tar_relative_pos_upper = np.array([0.0, 0.0, 0.20])
        self.tar_relative_pos_lower = np.array([0.0, 0.0, 0.20])
        self.tar_relative_rot_upper = np.array([0, 0, 0])
        self.tar_relative_rot_lower = np.array([0, 0, 0])

        self.relative_pos_upper = np.array([0.0, 0.0, 0.4])
        self.relative_pos_lower = np.array([0.0, 0.0, 0.5])
        self.relative_rot_upper = np.array([0.5, 0.5, np.pi/6])
        self.relative_rot_lower = np.array([-0.5,-0.5, -np.pi/6])

        # in_hand_error_pose
        # TODO test
        self.hand_error_pos_upper = np.array([0.0, 0.0, 0.2])
        self.hand_error_pos_lower = np.array([0.0, 0.0, 0.2])
        # self.hand_error_rot_upper = np.array([np.pi, 0.0, 0.0])
        # self.hand_error_rot_lower = np.array([np.pi, 0.0, 0.0])
        self.hand_error_rot_upper = np.array([np.pi + np.pi / 12, np.pi / 12, np.pi])
        self.hand_error_rot_lower = np.array([np.pi - np.pi / 12, -np.pi / 12, -np.pi])

        # self.hand_error_pos_upper = np.array([0.02, 0.0, 0.14])
        # self.hand_error_pos_lower = np.array([-0.02, 0.0, 0.12])
        # self.hand_error_rot_upper = np.array([0.0, 0.0, -3.14])
        # self.hand_error_rot_lower = np.array([0.0, 0.3, 3.14])

        self.render = render

        self.world = World(physics_dt=physics_dt, stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()

        hole_asset_path = os.path.join(
            root_path, "assets/pocketbook_pro_602_obj/pad.usd"
        )
        peg_asset_path = os.path.join(root_path, "assets/iphone/iphone.usd")

        print("peg_asset_path: ", peg_asset_path)
        print("hole_asset_path: ", hole_asset_path)
        self.camera_to_end_effector_pos = np.array([0.0, 0.1, 0.0])
        self.camera_to_end_effector_ori = R.from_euler(
            "xzx", [np.pi, np.pi / 2, np.pi / 2 + np.pi / 15]
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
        self.policy = IBVS(use_median_depth=False)
        self.pinhole_camera = PinholeCamera(
                        width=640,
                        height=480,
                        fx=616.56402588,
                        fy=616.59606934,
                        cx=330.48983765,
                        cy=233.84162903,
                    )
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
        self.camera.add_distance_to_image_plane_to_frame()

    def reset(self):
        self.count = 0

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
        _, _, tar_relative_T = samplePose(
            self.tar_relative_pos_upper,
            self.tar_relative_pos_lower,
            self.tar_relative_rot_upper,
            self.tar_relative_rot_lower,
        )
        _, _, relative_T = samplePose(
            self.relative_pos_upper,
            self.relative_pos_lower,
            self.relative_rot_upper,
            self.relative_rot_lower,
        )
        relative_pos = tar_relative_T[:3, 3]
        relative_q = tf.rot_matrix_to_quat(tar_relative_T[:3, :3])
        self.tar_peg_T = hole_T @ default_T @ tar_relative_T
        self.init_peg_T = hole_T @ default_T @ relative_T           
        self.peg_T = self.tar_peg_T
        self.set_peg_pose()

    def set_peg_pose(self):
        self.peg_pos = self.peg_T[:3, 3]
        self.peg_q = tf.rot_matrix_to_quat(self.peg_T[:3, :3])
        self.peg.set_world_pose(self.peg_pos, self.peg_q)

    def sample_hand_error(self):
        hand_error_pos, hand_error_ori, self.hand_error_T = samplePose(
            self.hand_error_pos_upper,
            self.hand_error_pos_lower,
            self.hand_error_rot_upper,
            self.hand_error_rot_lower,
        )

    def set_camera_pose(self):

        gripper_T = self.peg_T @ self.hand_error_T
        self.camera_T = gripper_T @ self.camera_to_end_effector_trans_matrix
        camera_q = tf.rot_matrix_to_quat(self.camera_T[:3, :3])
        camera_pos = self.camera_T[:3, 3]
        self.camera.set_world_pose(camera_pos, camera_q)
        
        
    
    def run(self):
        # self.model.test()
        self.count += 1
        if self.eva_count > 200:
            print(f"success rate: {self.success_count / self.eva_count}")
            print(f"rot error: {self.total_rot_error / self.success_count}")
            print(f"pos error: {self.total_pos_error / self.success_count}")
            exit()
        self.world.step(render=self.render)
        if self.world.is_playing():
            if self.world.current_time_step_index == 0:
                self.reset()
            else:
                if self.count == 1:
                    self.sample_hand_error()
                self.set_peg_pose()
                self.set_camera_pose()
                if self.count == 20 :
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
                        self.mask = segment_data["data"] == segment_id["peg"]
                        
                        seg_img[self.mask] = img[self.mask]
                        np.save("/home/chenyuxi/project/img-to-img-diffusion/test_img/mask.npy", self.mask)
                        cv2.imwrite("/home/chenyuxi/project/img-to-img-diffusion/test_img/seg_img.png", seg_img)
                        cv2.imwrite("/home/chenyuxi/project/img-to-img-diffusion/test_img/img.png", img)
                        pil_img = Image.fromarray(cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        model = self.load_model()
                        self.img_process = torchvision.transforms.Compose(
                            [
                                torchvision.transforms.Resize((240, 320)),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                            ]
                        )
                        self.reversed_img_process = torchvision.transforms.Resize((480, 640))
                        cv2.imwrite("/home/chenyuxi/project/img-to-img-diffusion/test_img/img.png", img)
                        cond_img = self.img_process(pil_img).to("cuda:1")
                        cond_img = cond_img.unsqueeze(0)
                        self.ref_img = model.eval(cond_img)
                        self.ref_img = np.transpose(self.reversed_img_process(self.ref_img).squeeze(0).numpy(), (1,2,0))
                        self.ref_img = ((self.ref_img + 1) * 127.5).round().astype(np.uint8)
                        Image.fromarray(self.ref_img).save("/home/chenyuxi/project/img-to-img-diffusion/test_img/ref_img.png")
                        tar_img = self.ref_img
                        # tar_img = img
                        tar_img[self.mask] = 0
                        tar_depth = self.camera.get_depth() 
                        tar_frame = Frame(
                            camera=self.pinhole_camera,
                            color=tar_img,
                            depth=tar_depth,
                            seg=None,
                            wcT=np.eye(4)
                        )
                        self.policy.set_desired(tar_frame)
                        self.peg_T = self.init_peg_T
                    else:
                        self.count = 0
                        self.sample_hand_error()
                        self.set_camera_pose()
                elif self.count > 30:
                    
                    cur_img = self.camera.get_rgba()[:, :, :3]
                    cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
                    # cur_pil_img = Image.fromarray(cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB))
                    # cur_img = self.img_process(cur_pil_img)
                    # cur_img = np.transpose(self.reversed_img_process(cur_img).numpy(), (1,2,0))
                    # cur_img = ((cur_img + 1) * 127.5).round().astype(np.uint8)
                    depth_img = self.camera.get_depth()
                    cur_img[self.mask] = 0
                    cur_frame = Frame(
                        camera=self.pinhole_camera,
                        color=cur_img,
                        depth=depth_img,
                        seg=None,
                        wcT=np.eye(4)
                    )
                    try:
                        vel, aux = self.policy.compute_velocity(cur_frame)
                    except:
                        print("Visual Servo Failed!!")
                        self.eva_count += 1
                        self.count = 0
                        self.peg_T = self.tar_peg_T
                        return
                    trans_vel = np.zeros(6)
                    trans_vel[0] = vel[2]
                    trans_vel[1] = -1 * vel[0]
                    trans_vel[2] = -1 * vel[1]
                    trans_vel[3] = vel[5]
                    trans_vel[4] = -1 * vel[3]
                    trans_vel[5] = -1 * vel[4]
                    
                    # cv2.imshow("current image", cur_img)
                    cv2.imshow("desired image", depth_img)
                    cv2.imshow("desired | current", aux["plottings"])
                    cv2.waitKey(1)
                    # next_camera_T = action.action(self.camera_T, vel, 1/30)
                    next_camera_T = tf.calc_pose_from_vel(self.camera_T, trans_vel, 1/30)
                    # print(next_camera_T)
                    self.peg_T = next_camera_T @ np.linalg.inv(self.camera_to_end_effector_trans_matrix) @ np.linalg.inv(self.hand_error_T)
                    error = self.peg_T @ np.linalg.inv(self.tar_peg_T)
                    pos_error = np.linalg.norm(error[:3, 3]) * 1000.0
                    rot = R.from_matrix(error[:3, :3]).as_rotvec()
                    rot_error = np.linalg.norm(rot) / np.pi * 180
                    # print("pos error: ", pos_error)
                    # print("rot error: ", rot_error)
                    # self.set_peg_pose()
                    # self.set_camera_pose()
                    # print(self.count)
                
                if self.count > 300:
                    self.count = 0
                    error = self.peg_T @ np.linalg.inv(self.tar_peg_T)
                    pos_error = np.linalg.norm(error[:3, 3]) * 1000.0
                    rot = R.from_matrix(error[:3, :3]).as_rotvec()
                    rot_error = np.linalg.norm(rot) / np.pi * 180
                    if pos_error < 20.0 and rot_error < 10:
                        self.success_count += 1
                        self.total_pos_error += pos_error
                        self.total_rot_error += rot_error
                        print(f"rot error: {self.total_rot_error / self.success_count}")
                        print(f"pos error: {self.total_pos_error / self.success_count}")
                    self.eva_count += 1
                    print(f"success rate: {self.success_count / self.eva_count}")
                    self.peg_T = self.tar_peg_T
                    
                        
                    
                    
                                        
                    
                
    def load_model(self):
        print(torch.cuda.current_device())
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-c",
            "--config",
            type=str,
            default="config/vs.json",
            help="JSON file for configuration",
        )
        parser.add_argument(
            "-p",
            "--phase",
            type=str,
            choices=["train", "test"],
            help="Run train or test",
            default="train",
        )
        parser.add_argument(
            "-b", "--batch", type=int, default=None, help="Batch size in every gpu"
        )
        parser.add_argument("-gpu", "--gpu_ids", type=str, default=None)
        parser.add_argument("-d", "--debug", action="store_true")
        parser.add_argument("-P", "--port", default="21012", type=str)

        """ parser configs """
        args = parser.parse_args()
        opt = Praser.parse(args)

        """ cuda devices """
        gpu_str = ",".join(str(x) for x in opt["gpu_ids"])

        # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
        if len(opt["gpu_ids"]) == 1:
            torch.cuda.set_device(opt["gpu_ids"][0])
        
        print("export CUDA_VISIBLE_DEVICES={}".format(gpu_str))
        """ use DistributedDataParallel(DDP) and multiprocessing for multi-gpu training"""
        # [Todo]: multi GPU on multi machine
        if opt["distributed"]:
            ngpus_per_node = len(opt["gpu_ids"])  # or torch.cuda.device_count()
            opt["world_size"] = ngpus_per_node
            opt["init_method"] = "tcp://127.0.0.1:" + args.port
            mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
        else:
            opt["world_size"] = 1
            return main_worker(0, 1, opt)
            

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