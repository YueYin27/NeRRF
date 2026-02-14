import imageio
import numpy as np
import cv2
import json
import math
import os
import sys
from os.path import join
from PIL import Image

sys.path.append("..")
import dataset.utils as api_utils
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path


def get_image_to_tensor_balanced(image_size):
    ops = []
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    ops.append(transforms.Resize(image_size))
    return transforms.Compose(ops)


def get_focal(meta_path, image_size, dataset_type):
    W, H = image_size
    camera_pose = np.eye(4, dtype=np.float32)
    with open(meta_path) as f:
        meta_data = json.load(f)
        camera_pose = np.array(meta_data["cam_world_pose"])
        if dataset_type == "blender":
            dx = math.radians(60)
            fx = (W / 2) / math.tan(dx / 2)
            fy = fx
        elif dataset_type == "eikonal":
            fx = fy = meta_data["f"]
        else:
            raise NotImplementedError
    camera_pose[..., 3] = camera_pose[..., 3]
    return fx, fy, camera_pose


def get_mvp(focal, pose, image_size, far=50, near=0.001):
    W, H = image_size
    projection = np.array(
        [
            [2 * focal / W, 0, 0, 0],
            [0, -2 * focal / H, 0, 0],
            [0, 0, -(far + near) / (far - near), -(2 * far * near) / (far - near)],
            [0, 0, -1, 0],
        ],
        dtype=np.float32,
    )
    return projection @ np.linalg.inv(pose)  # [4, 4]


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        stage="train",
        dataset_type="blender",  # if use eikonal dataset, change this to "eikonal"
    ):
        super().__init__()
        self.split = stage
        self.type = dataset_type
        if self.type == "blender":
            self.image_size = (800, 800)
            self.z_near, self.z_far = 0.05, 1e5
            # self.depth_dir = data_dir + "/../../depth/" + os.path.basename(data_dir)
            if self.split == "train":
                self.image_dir = data_dir + '/train'
                self.meta_dir = data_dir + "/transforms_train.json"
            else:
                self.image_dir = data_dir + '/test'
                self.meta_dir = data_dir + "/transforms_test.json"
                self.image_dir = data_dir + '/train'
                self.meta_dir = data_dir + "/transforms_train.json"

            self.image_list = list(sorted(
                [p for p in Path(self.image_dir).glob('r_*.png') if p.stem.count('_') == 1],
                key=lambda p: int(p.stem.split('_')[1])
            ))
        else:
            raise NotImplementedError
        
        with Path(self.meta_dir).open('r') as f:
            self.meta = json.load(f)
        
        self.focal = float(
            0.5 * self.image_size[1] / np.tan(0.5 * self.meta["camera_angle_x"])
        )

        self.image2tensor = get_image_to_tensor_balanced(
            (self.image_size[1], self.image_size[0])
        )
        name, focal, poses, images, mvps, masks = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for i in range(len(self.image_list)):
            result = self.__get_single_item__(i)
            name.append(result["name"])
            focal.append(result["focal"])
            poses.append(result["poses"])
            images.append(result["images"])
            mvps.append(result["mvp"])
            masks.append(result["mask"])

        results = {
            "name": name,
            "focal": torch.stack(focal),
            "poses": torch.stack(poses),
            "images": torch.stack(images),
            "mvp": torch.stack(mvps),
            "mask": torch.stack(masks),
        }
        self.results = results

    def __len__(self):
        return len(self.image_list)

    def __get_single_item__(self, index):
        img_path = self.image_list[index]
        name = img_path.stem
        if self.type == "blender":
            img_name = img_path.name
            # meta_name = name + "_meta_0000.json"
        else:
            raise NotImplementedError

        image_path = str(img_path)
        img = imageio.imread(image_path)

        if img.shape[-1] == 4:
            # RGBA image: extract alpha as mask, composite over white
            alpha = img[..., 3:4].astype(np.float32) / 255.0
            rgb = img[..., :3].astype(np.float32) / 255.0
            mask = alpha[..., 0] > 0.01
            img = (rgb * alpha + (1.0 - alpha)) * 255.0
        else:
            # RGB image: load separate mask file
            mask_path = img_path.parent / f"{name}_mask_0000.png"
            if mask_path.exists():
                mask_img = imageio.imread(str(mask_path))
                if mask_img.ndim == 3:
                    mask_img = mask_img[..., 0]
                mask = (mask_img / 255.) > 0.01
            else:
                mask = np.ones(img.shape[:2], dtype=bool)
            img = img[..., :3]

        mask = torch.tensor(mask).unsqueeze(0)
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = self.image2tensor(img)

        fx = fy = self.focal
        camera_pose = self.meta["frames"][index]["transform_matrix"] # FIXME

        mvp = get_mvp(fx, camera_pose, self.image_size)
        result = {
            "name": name,
            "focal": torch.tensor((fx, fy), dtype=torch.float32),
            "poses": torch.tensor(camera_pose, dtype=torch.float32),
            "images": img,
            "mvp": torch.tensor(mvp, dtype=torch.float32),
            "mask": mask,
        }
        return result

    def __getitem__(self, index):
        result = {
            "name": self.results["name"][index],
            "focal": self.results["focal"][index],
            "poses": self.results["poses"][index],
            "images": self.results["images"][index],
            "mask": self.results["mask"][index],
            "mvp": self.results["mvp"][index],
        }
        return result
