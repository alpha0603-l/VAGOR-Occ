import os
import torch
import numpy as np
from numpy import random
import mmcv
from PIL import Image
import math

from . import OPENOCC_TRANSFORMS

from .loading_utils import load_augmented_point_cloud, reduce_LiDAR_beams
# from mmdet3d.core.points import BasePoints, get_points_type
from mmdet3d.structures.points import BasePoints, get_points_type
import mmengine

"""
A series of instances from the Registry class: OPENOCC_TRANSFORMS
"""

@OPENOCC_TRANSFORMS.register_module()
class DefaultFormatBundle(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """

    def __init__(self, ):
        return

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # If the key is not in the results, skip it.
        if 'img' in results:
            if isinstance(results['img'], list):
                # process multiple imgs in single frame
                # original shape: [H, W, C]
                # new shape: [C, H, W]
                imgs = [img.transpose(2, 0, 1) for img in results['img']]
                # shape: [N, C, H, W]
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
            else:
                imgs = np.ascontiguousarray(results['img'].transpose(2, 0, 1))
            # transform to tensor and with shape of [N, C, H, W]
            results['img'] = torch.from_numpy(imgs)

        if 'points' in results:
            assert isinstance(results["points"], BasePoints)
            results["points"] = results["points"].tensor
            # results["points"] = torch.from_numpy(results["points"])

        # Optional full point tensor for visibility / octree modules.
        # This is intentionally separated from ``points`` so the LiDAR encoder
        # can consume 6-D features while visibility keeps the 7-D current flag.
        if 'visibility_points' in results:
            assert isinstance(results["visibility_points"], BasePoints)
            results["visibility_points"] = results["visibility_points"].tensor

        if 'lidar_feature_maps' in results:
            assert isinstance(results["lidar_feature_maps"], dict)
            for key, value in results["lidar_feature_maps"].items():
                assert isinstance(value, np.ndarray)
                results["lidar_feature_maps"][key] = torch.from_numpy(value)

        if 'dpt' in results:
            if isinstance(results['dpt'], list):
                dpts = results['dpt'] # list of (H, W)
                # (N, 1, H, W)
                dpts = np.ascontiguousarray(np.stack(dpts, axis=0))
                if len(dpts.shape) == 2:
                    dpts = dpts[None, :, :]
                dpts = dpts[:, None, :, :]
            else:
                dpts = np.ascontiguousarray(results['dpt'][None,:,:])
            # from numpy to tensor
            results['dpt'] = torch.from_numpy(dpts) # (N, 1, H, W)
        
        if 'anchor_points' in results:
            assert isinstance(results["anchor_points"], np.ndarray)
            results["anchor_points"] = torch.from_numpy(results["anchor_points"])

        if 'occ3d_mask_camera' in results:
            assert isinstance(results["occ3d_mask_camera"], np.ndarray)
            results["occ3d_mask_camera"] = torch.from_numpy(results["occ3d_mask_camera"])
        
        return results

    def __repr__(self):
        return self.__class__.__name__


@OPENOCC_TRANSFORMS.register_module()
class NuScenesAdaptor(object):
    def __init__(self, num_cams, use_ego=False):
        self.num_cams = num_cams
        self.projection_key = 'ego2img' if use_ego else 'lidar2img'
        self.T_key = 'ego2global' if use_ego else 'lidar2global'
        pass

    def __call__(self, input_dict):
        input_dict["projection_mat"] = np.float32(
            np.stack(input_dict[self.projection_key])
        )
        if len(input_dict["projection_mat"].shape) == 2:
            input_dict["projection_mat"] = input_dict["projection_mat"][None, :, :]
        input_dict["image_wh"] = np.ascontiguousarray(
            np.array(input_dict["img_shape"], dtype=np.float32)[:, :2][:, ::-1]
        )
        # input_dict["T_global_inv"] = np.linalg.inv(input_dict[self.T_key])
        # input_dict["T_global"] = input_dict[self.T_key]
        # if "cam_intrinsic" in input_dict:
        #     input_dict["cam_intrinsic"] = np.float32(
        #         np.stack(input_dict["cam_intrinsic"]))
        #     input_dict["focal"] = input_dict["cam_intrinsic"][..., 0, 0]
        # input_dict["extrinsics"] = input_dict["lidar2temCam"]
        # input_dict["intrinsics"] = input_dict["ori_intrinsic"][..., :3, :3]
        return input_dict


@OPENOCC_TRANSFORMS.register_module()
class ResizeCropFlipImage(object):
    def __call__(self, results):
        aug_configs = results.get("aug_configs")
        if aug_configs is None:
            return results
        resize, resize_dims, crop, flip, rotate = aug_configs # aug_configs = [resize, resize_dims, crop, flip, rotate]
        imgs = results["img"]
        N = len(imgs) # N: number of views
        new_imgs = []
        if 'dpt' in results:
            dpts = results['dpt']
            new_dpts = []
        for i in range(N):
            # here imgs[i] is in shape of [H, W, C]
            # data is always in [H, W, C] format but Image processes them in [W, H, C] format!!
            img = Image.fromarray(np.uint8(imgs[i]))
            img, ida_mat = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            if 'dpt' in results:
                dpt = dpts[i]
                # Depth maps must be float32 before PIL mode='F'. Passing float64 with
                # a forced 'F' mode can corrupt the underlying bytes and produce huge
                # invalid values, which makes depth loss become exactly zero.
                dpt = np.asarray(dpt, dtype=np.float32)
                dpt[~np.isfinite(dpt)] = 0.0
                dpt = Image.fromarray(dpt, mode='F')
                dpt, _ = self._img_transform(
                    dpt,
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate,
                    resample=Image.NEAREST,
                )
                dpt = np.array(dpt).astype(np.float32)
                dpt[~np.isfinite(dpt)] = 0.0
                dpt[dpt < 0.0] = 0.0
                new_dpts.append(dpt)

            mat = np.eye(4)
            mat[:3, :3] = ida_mat # Rotation matrix
            # Store the transformed image to new_imgs
            new_imgs.append(np.array(img).astype(np.float32))
            results["lidar2img"][i] = mat @ results["lidar2img"][i]
            results["ego2img"][i] = mat @ results["ego2img"][i]
            if "cam_intrinsic" in results:
                results["cam_intrinsic"][i][:3, :3] *= resize

        # Update the results with the new images
        results["img"] = new_imgs
        if 'dpt' in results:
            results['dpt'] = new_dpts
        results["img_shape"] = [x.shape[:2] for x in new_imgs]
        return results

    def _get_rot(self, h):
        return torch.Tensor(
            [
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ]
        )

    def _img_transform(self, img, resize, resize_dims, crop, flip, rotate, resample=None):
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        # adjust image. Keep the old PIL default for RGB images; use explicit
        # nearest-neighbor for sparse depth maps to avoid interpolating invalid zeros.
        if resample is None:
            img = img.resize(resize_dims)
        else:
            img = img.resize(resize_dims, resample=resample)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        if resample is None:
            img = img.rotate(rotate)
        else:
            img = img.rotate(rotate, resample=resample)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self._get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = torch.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat


@OPENOCC_TRANSFORMS.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results["img"] = [
            mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
            for img in results["img"]
        ]
        results["img_norm_cfg"] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb
        )
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})"
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(
        self,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results["img"]
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, (
                "PhotoMetricDistortion needs the input image of dtype np.float32,"
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            )
            # random brightness
            if random.randint(2):
                delta = random.uniform(
                    -self.brightness_delta, self.brightness_delta
                )
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(
                        self.contrast_lower, self.contrast_upper
                    )
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(
                    self.saturation_lower, self.saturation_upper
                )

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(
                        self.contrast_lower, self.contrast_upper
                    )
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        # Update new image list
        results["img"] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(\nbrightness_delta={self.brightness_delta},\n"
        repr_str += "contrast_range="
        repr_str += f"{(self.contrast_lower, self.contrast_upper)},\n"
        repr_str += "saturation_range="
        repr_str += f"{(self.saturation_lower, self.saturation_upper)},\n"
        repr_str += f"hue_delta={self.hue_delta})"
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class LoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool, optional): Whether to convert the img to float32.
            Defaults to False.
        color_type (str, optional): Color type of the file.
            Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formatting.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])] # unpack multi-view images to list
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class LoadPointFromFileLiDAR(object):
    """Load LiDAR Points From File.

    Load sunrgbd and scannet points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
    """

    def __init__(
        self,
        coord_type,
        load_dim=6, # for our case: 5
        use_dim=[0, 1, 2], # for our case: 5
        shift_height=False,
        use_color=False,
        load_augmented=None,
        reduce_beams=None,
    ):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert (
            max(use_dim) < load_dim
        ), f"Expect all used dimensions < {load_dim}, got {use_dim}"
        assert coord_type in ["CAMERA", "LIDAR", "DEPTH"]

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.load_augmented = load_augmented
        self.reduce_beams = reduce_beams

    def _load_points(self, lidar_path):
        """Private function to load point clouds data.

        Args:
            lidar_path (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        # mmcv.check_file_exist(lidar_path)
        mmengine.check_file_exist(lidar_path)
        if self.load_augmented:
            assert self.load_augmented in ["pointpainting", "mvp"]
            virtual = self.load_augmented == "mvp"
            points = load_augmented_point_cloud(
                lidar_path, virtual=virtual, reduce_beams=self.reduce_beams
            )
        elif lidar_path.endswith(".npy"):
            points = np.load(lidar_path)
        else:
            points = np.fromfile(lidar_path, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        lidar_path = results["lidar_path"]
        # lidar_path = results["pts_filename"]
        points = self._load_points(lidar_path)
        points = points.reshape(-1, self.load_dim)
        # check reduced beams
        if self.reduce_beams and self.reduce_beams < 32:
            points = reduce_LiDAR_beams(points, self.reduce_beams)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3], np.expand_dims(height, 1), points[:, 3:]], 1
            )
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(
                    color=[
                        points.shape[1] - 3,
                        points.shape[1] - 2,
                        points.shape[1] - 1,
                    ]
                )
            )

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims
        )
        results["points"] = points

        return results
    
    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class LoadPointsFromMultiSweepsLiDAR(object):
    """Load and aggregate nuScenes multi-sweep LiDAR points.

    This version follows the DAOcc-style loading logic used in the user's
    previous Gaussian branch:

        raw 5-D point:       [x, y, z, intensity, time_lag]
        model 6-D point:     [x, y, z, intensity, time_lag, decay]
        visibility 7-D point:[x, y, z, intensity, time_lag, decay, is_current]

    ``results['points']`` is kept as the model input for the LiDAR encoder.
    ``results['visibility_points']`` is kept only for visibility / octree /
    measured-Gaussian generation. This avoids feeding the hard binary
    ``is_current`` flag into the LiDAR encoder while still preserving it for
    strong/weak free-space reasoning.
    """

    def __init__(
        self,
        sweeps_num=10,
        load_dim=5,
        use_dim=6,
        visibility_use_dim=7,
        pad_empty_sweeps=False,
        remove_close=False,
        test_mode=False,
        load_augmented=None,
        reduce_beams=None,
        add_decay=True,
        decay_lambda=1.0,
        decay_normalize=True,
        add_is_current=True,
        point_cloud_range=None,
        crop_points=False,
        sort_sweeps_by_time=True,
        debug=False,
        debug_print_interval=20,
        debug_max_print=5,
    ):
        self.load_dim = int(load_dim)
        self.sweeps_num = int(sweeps_num)

        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        if isinstance(visibility_use_dim, int):
            visibility_use_dim = list(range(visibility_use_dim))
        self.use_dim = list(use_dim)
        self.visibility_use_dim = list(visibility_use_dim)

        self.pad_empty_sweeps = bool(pad_empty_sweeps)
        self.remove_close = bool(remove_close)
        self.test_mode = bool(test_mode)
        self.load_augmented = load_augmented
        self.reduce_beams = reduce_beams

        self.add_decay = bool(add_decay)
        self.decay_lambda = float(decay_lambda)
        self.decay_normalize = bool(decay_normalize)
        self.add_is_current = bool(add_is_current)
        self.time_lag_dim = 4

        if point_cloud_range is None:
            self.point_cloud_range = None
        else:
            assert len(point_cloud_range) == 6, (
                "point_cloud_range should be "
                "[x_min, y_min, z_min, x_max, y_max, z_max], "
                f"got {point_cloud_range}"
            )
            self.point_cloud_range = [float(v) for v in point_cloud_range]
        self.crop_points = bool(crop_points)
        self.sort_sweeps_by_time = bool(sort_sweeps_by_time)

        self.debug = bool(debug)
        self.debug_print_interval = max(int(debug_print_interval), 1)
        self.debug_max_print = max(int(debug_max_print), 0)
        self._debug_seen_count = 0
        self._debug_print_count = 0

        assert self.load_dim > self.time_lag_dim, (
            f"LoadPointsFromMultiSweepsLiDAR expects load_dim > {self.time_lag_dim}, "
            f"got load_dim={self.load_dim}"
        )

        max_available_dim = (
            self.load_dim
            + (1 if self.add_decay else 0)
            + (1 if self.add_is_current else 0)
        )
        if len(self.use_dim) > 0:
            assert max(self.use_dim) < max_available_dim, (
                f"use_dim={self.use_dim} exceeds available dims {max_available_dim}."
            )
        if len(self.visibility_use_dim) > 0:
            assert max(self.visibility_use_dim) < max_available_dim, (
                f"visibility_use_dim={self.visibility_use_dim} exceeds available dims "
                f"{max_available_dim}."
            )

    def _load_points(self, lidar_path):
        """Private function to load point clouds data."""
        mmengine.check_file_exist(lidar_path)
        if self.load_augmented:
            assert self.load_augmented in ["pointpainting", "mvp"]
            virtual = self.load_augmented == "mvp"
            points = load_augmented_point_cloud(
                lidar_path, virtual=virtual, reduce_beams=self.reduce_beams
            )
        elif lidar_path.endswith(".npy"):
            points = np.load(lidar_path)
        else:
            points = np.fromfile(lidar_path, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Remove points too close to the LiDAR origin."""
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.detach().cpu().numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def _crop_points_by_range(self, points):
        """Crop ndarray/BasePoints by self.point_cloud_range if enabled."""
        if self.point_cloud_range is None or not self.crop_points:
            return points

        x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range
        if isinstance(points, np.ndarray):
            pts = points
            mask = (
                (pts[:, 0] >= x_min) & (pts[:, 0] <= x_max) &
                (pts[:, 1] >= y_min) & (pts[:, 1] <= y_max) &
                (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
            )
            return points[mask]

        if isinstance(points, BasePoints):
            pts = points.tensor
            mask = (
                (pts[:, 0] >= x_min) & (pts[:, 0] <= x_max) &
                (pts[:, 1] >= y_min) & (pts[:, 1] <= y_max) &
                (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
            )
            return points[mask]

        raise NotImplementedError

    @staticmethod
    def _new_points_with_dim(points, points_tensor):
        return points.__class__(
            points_tensor,
            points_dim=points_tensor.shape[-1],
            attribute_dims=getattr(points, "attribute_dims", None),
        )

    @staticmethod
    def _timestamp_to_sec(timestamp):
        """Convert common nuScenes timestamp formats to seconds.

        In this project, ``results['timestamp']`` is already converted to
        seconds by the dataset, while sweep timestamps may still be stored as
        raw nuScenes microseconds.  Dividing both by 1e6 again produces the
        observed huge negative time lag, e.g. -1.53e9.
        """
        if timestamp is None:
            return None
        if isinstance(timestamp, torch.Tensor):
            timestamp = timestamp.detach().cpu().item()
        elif isinstance(timestamp, np.ndarray):
            timestamp = float(np.asarray(timestamp).reshape(-1)[0])
        else:
            timestamp = float(timestamp)

        abs_ts = abs(timestamp)
        # nuScenes raw timestamp is usually microseconds, about 1.5e15.
        if abs_ts > 1e14:
            return timestamp / 1e6
        # Some intermediate infos store milliseconds, about 1.5e12.
        if abs_ts > 1e11:
            return timestamp / 1e3
        # Dataset already returns seconds, about 1.5e9, or a relative time.
        return timestamp

    def _append_decay_feature(self, points):
        """Compute temporal decay from the absolute time_lag channel."""
        time_lag = points.tensor[:, self.time_lag_dim:self.time_lag_dim + 1].float()
        # Use absolute lag so the decay is robust even if an upstream info file
        # stores sweep lag with the opposite sign.
        time_lag = torch.abs(time_lag)

        if self.decay_normalize:
            max_lag = torch.max(time_lag)
            time_for_decay = time_lag / torch.clamp(max_lag, min=1e-6)
        else:
            time_for_decay = time_lag

        decay = torch.exp(-self.decay_lambda * time_for_decay)
        return decay.to(dtype=points.tensor.dtype, device=points.tensor.device)

    def _debug_output(self, results, debug_info):
        if not self.debug:
            return
        self._debug_seen_count += 1
        debug_info["debug_seen_count"] = int(self._debug_seen_count)
        results["multi_sweep_decay_current_debug"] = debug_info
        if (
            self._debug_print_count < self.debug_max_print
            and self._debug_seen_count % self.debug_print_interval == 0
        ):
            print("[LoadPointsFromMultiSweepsLiDAR 6D+7D Debug]", debug_info)
            self._debug_print_count += 1

    def __call__(self, results):
        """Load multi-sweep point clouds and build model/visibility tensors."""
        points = results["points"]
        if not isinstance(points, BasePoints):
            raise TypeError(f"results['points'] must be BasePoints, got {type(points)}")
        if points.tensor.shape[1] < self.load_dim:
            raise ValueError(
                f"Current points dim {points.tensor.shape[1]} < load_dim={self.load_dim}"
            )

        # Keep the raw 5-D base first: [x, y, z, intensity, time_lag].
        # Current frame time_lag is always zero.
        points = self._new_points_with_dim(points, points.tensor[:, :self.load_dim].contiguous())
        points.tensor[:, self.time_lag_dim] = 0
        points = self._crop_points_by_range(points)
        num_current_points = int(points.tensor.shape[0])
        sweep_points_list = [points]

        current_ts = self._timestamp_to_sec(results.get("timestamp", None))
        if current_ts is None:
            raise KeyError("results must contain a valid 'timestamp' for multi-sweep time_lag/decay")

        sweeps = results.get("sweeps", [])
        if self.sort_sweeps_by_time and len(sweeps) > 0:
            sweeps = sorted(
                sweeps,
                key=lambda x: self._timestamp_to_sec(x.get("timestamp", 0.0)),
                reverse=True,
            )

        num_available_sweeps = len(sweeps)
        num_selected_sweeps = 0
        selected_sweep_time_lags = []

        if self.pad_empty_sweeps and num_available_sweeps == 0:
            for _ in range(self.sweeps_num):
                if self.remove_close:
                    points_pad = self._remove_close(points)
                else:
                    points_pad = points
                points_pad_tensor = points_pad.tensor.clone()
                points_pad_tensor[:, self.time_lag_dim] = 0
                points_pad = self._new_points_with_dim(points_pad, points_pad_tensor)
                sweep_points_list.append(points_pad)
            num_selected_sweeps = self.sweeps_num
        else:
            if num_available_sweeps <= self.sweeps_num:
                choices = np.arange(num_available_sweeps)
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                if not self.load_augmented:
                    choices = np.random.choice(num_available_sweeps, self.sweeps_num, replace=False)
                else:
                    choices = np.random.choice(num_available_sweeps - 1, self.sweeps_num, replace=False)
                choices = np.sort(choices)

            num_selected_sweeps = int(len(choices))
            for idx in choices:
                sweep = sweeps[int(idx)]
                points_sweep = self._load_points(sweep["data_path"])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)

                if self.reduce_beams and self.reduce_beams < 32:
                    points_sweep = reduce_LiDAR_beams(points_sweep, self.reduce_beams)

                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)

                sweep_ts = self._timestamp_to_sec(sweep.get("timestamp", None))
                if sweep_ts is None:
                    raise KeyError("Each sweep must contain a valid 'timestamp' for time_lag/decay")

                points_sweep[:, :3] = points_sweep[:, :3] @ sweep["sensor2lidar_rotation"].T
                points_sweep[:, :3] += sweep["sensor2lidar_translation"]

                # Store positive seconds elapsed from the historical sweep to
                # the current keyframe.  Current frame stays 0.
                time_lag = max(float(current_ts - sweep_ts), 0.0)
                points_sweep[:, self.time_lag_dim] = time_lag
                selected_sweep_time_lags.append(time_lag)

                points_sweep = self._crop_points_by_range(points_sweep)

                points_sweep = self._new_points_with_dim(
                    points,
                    points.tensor.new_tensor(points_sweep),
                )
                sweep_points_list.append(points_sweep)

        points_all = points.cat(sweep_points_list)
        points_tensor_before_feature = points_all.tensor

        feature_list = [points_all.tensor[:, :self.load_dim]]

        decay = None
        if self.add_decay:
            decay = self._append_decay_feature(points_all)
            feature_list.append(decay)

        is_current = None
        if self.add_is_current:
            is_current = points_all.tensor.new_zeros((points_all.tensor.shape[0], 1))
            if num_current_points > 0:
                is_current[:num_current_points, 0] = 1
            feature_list.append(is_current)

        full_tensor = torch.cat(feature_list, dim=1)
        full_points = self._new_points_with_dim(points_all, full_tensor)

        # 6-D tensor for LiDAR encoder: [x, y, z, intensity, time_lag, decay].
        results["points"] = full_points[:, self.use_dim]

        # 7-D tensor for visibility/octree only: [x, y, z, intensity, time_lag, decay, is_current].
        # Keep this separate so the hard current-frame flag does not directly enter the LiDAR encoder.
        results["visibility_points"] = full_points[:, self.visibility_use_dim]

        if self.debug:
            time_lags = points_tensor_before_feature[:, self.time_lag_dim]
            debug_info = {
                "shape_before_feature_build": tuple(points_tensor_before_feature.shape),
                "shape_full": tuple(full_tensor.shape),
                "shape_model_points": tuple(results["points"].tensor.shape),
                "shape_visibility_points": tuple(results["visibility_points"].tensor.shape),
                "sweeps_num_cfg": int(self.sweeps_num),
                "num_available_sweeps": int(num_available_sweeps),
                "num_selected_sweeps": int(num_selected_sweeps),
                "add_decay": bool(self.add_decay),
                "add_is_current": bool(self.add_is_current),
                "point_cloud_range": self.point_cloud_range,
                "crop_points": bool(self.crop_points),
                "decay_lambda": float(self.decay_lambda),
                "decay_normalize": bool(self.decay_normalize),
                "use_dim": list(self.use_dim),
                "visibility_use_dim": list(self.visibility_use_dim),
                "time_lag_min": float(time_lags.min().item()) if time_lags.numel() > 0 else 0.0,
                "time_lag_max": float(time_lags.max().item()) if time_lags.numel() > 0 else 0.0,
                "current_timestamp_sec": float(current_ts),
                "selected_sweep_time_lag_min": float(min(selected_sweep_time_lags)) if selected_sweep_time_lags else 0.0,
                "selected_sweep_time_lag_max": float(max(selected_sweep_time_lags)) if selected_sweep_time_lags else 0.0,
                "selected_sweep_time_lags_first10": [float(v) for v in selected_sweep_time_lags[:10]],
            }
            if decay is not None and decay.numel() > 0:
                debug_info.update({
                    "decay_min": float(decay.min().item()),
                    "decay_max": float(decay.max().item()),
                    "decay_mean": float(decay.mean().item()),
                })
            if is_current is not None:
                current_count = int((is_current[:, 0] > 0.5).sum().item())
                debug_info.update({
                    "is_current_count": current_count,
                    "history_or_padding_count": int(is_current.shape[0] - current_count),
                })
            self._debug_output(results, debug_info)

        return results

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(sweeps_num={self.sweeps_num}, "
            f"load_dim={self.load_dim}, use_dim={self.use_dim}, "
            f"visibility_use_dim={self.visibility_use_dim}, "
            f"add_decay={self.add_decay}, add_is_current={self.add_is_current})"
        )

@OPENOCC_TRANSFORMS.register_module()
class LoadMultiViewDepthFromFiles(object):
    """Load the gound truth depth map generated from BEVDepth using lidar.
    """

    def __init__(self, is_to_depth_map=True, map_size=None):
        self.is_to_depth_map = is_to_depth_map
        self.map_size = map_size

    def __call__(self, results):
        if self.map_size is None:
            map_size = tuple(results['img'][0].shape[:2])  # (H, W), usually (900, 1600)
        else:
            map_size = tuple(self.map_size)
        img_paths = results['img_filename']
        dpt_paths = []
        map_depths = []
        for img_path in img_paths:
            dpt_path = os.path.join(img_path.split("/samples/")[0], "depth_gt", img_path.split("/")[-1]+".bin")
            point_depth = np.fromfile(dpt_path, dtype=np.float32, count=-1).reshape(-1, 3)
            dpt_paths.append(dpt_path)
            if self.is_to_depth_map:
                map_depth = self.to_depth_map(point_depth, map_size)
                map_depths.append(map_depth)
            
        # img is of shape (h, w, c, num_views)
        # map_depths is a List of depth maps, each of shape (H, W)
        # N * (H, W), N = num_views
        results['dpt'] = map_depths
        results['filename_dpt'] = dpt_paths
        return results
    
    def to_depth_map(self, point_depth, map_size):
        """Transform depth based on ida augmentation configuration.

        Args:
            cam_depth (np array): Nx3, 3: x,y,d.
            resize (float): Resize factor.
            resize_dims (list): Final dimension.
            crop (list): x1, y1, x2, y2
            flip (bool): Whether to flip.
            rotate (float): Rotation value.

        Returns:
            np array: [h/down_ratio, w/down_ratio, d]
        """

        # Here they assume the point depth coordinates are 900, 1600?
        # TODO: check the point depth coordinate is (H, W) or (W, H)
        depth_coords = point_depth[:, :2].astype(np.int16)

        # The loaded image shape is also 900, 1600?
        depth_map = np.zeros(map_size, dtype=np.float32) # (H, W)
        valid_mask = ((depth_coords[:, 1] < map_size[0])
                    & (depth_coords[:, 0] < map_size[1])
                    & (depth_coords[:, 1] >= 0)
                    & (depth_coords[:, 0] >= 0))
        depth_map[depth_coords[valid_mask, 1],
                depth_coords[valid_mask, 0]] = point_depth[valid_mask, 2].astype(np.float32)

        depth_map[~np.isfinite(depth_map)] = 0.0
        depth_map[depth_map < 0.0] = 0.0
        return depth_map.astype(np.float32, copy=False)
    
    def __repr__(self):
        return self.__class__.__name__


@OPENOCC_TRANSFORMS.register_module()
class PadMultiViewImage(object):
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor # 32, TODO: choose (864, 1600), (896, 1600), (928, 1600)
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = [mmcv.impad(
                img, shape=self.size, pad_val=self.pad_val) for img in results['img']]
            if "dpt" in results.keys():
                padded_dpt = [mmcv.impad(
                img, shape=self.size, pad_val=self.pad_val) for img in results['dpt']]
        elif self.size_divisor is not None: # here, 32
            padded_img = [mmcv.impad_to_multiple(
                img, self.size_divisor, pad_val=self.pad_val) for img in results['img']]
            if "dpt" in results.keys():
                padded_dpt = [mmcv.impad_to_multiple(
                    img, self.size_divisor, pad_val=self.pad_val) for img in results['dpt']]
        
        results['ori_shape'] = [img.shape for img in results['img']] # keep track of original shape which should be (900, 1600, 3)
        results['img'] = padded_img # now should be (928, 1600, 3), list of np.array (H, W, C)
        if "dpt" in results.keys():
            results['dpt'] = padded_dpt # now should be (928, 1600), list of np.array (H, W)
        results['img_shape'] = [img.shape for img in padded_img] # padded version will go into image_wh key
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class LoadOccupancySurroundOcc(object):

    def __init__(self, occ_path, semantic=False, use_ego=False, use_sweeps=False):
        self.occ_path = occ_path
        self.semantic = semantic
        self.use_ego = use_ego
        assert semantic and (not use_ego)
        self.use_sweeps = use_sweeps

        xyz = self.get_meshgrid([-50, -50, -5.0, 50, 50, 3.0], [200, 200, 16], 0.5)
        self.xyz = np.concatenate([xyz, np.ones_like(xyz[..., :1])], axis=-1) # x, y, z, 4

    def get_meshgrid(self, ranges, grid, reso):
        xxx = torch.arange(grid[0], dtype=torch.float) * reso + 0.5 * reso + ranges[0]
        yyy = torch.arange(grid[1], dtype=torch.float) * reso + 0.5 * reso + ranges[1]
        zzz = torch.arange(grid[2], dtype=torch.float) * reso + 0.5 * reso + ranges[2]

        xxx = xxx[:, None, None].expand(*grid)
        yyy = yyy[None, :, None].expand(*grid)
        zzz = zzz[None, None, :].expand(*grid)

        xyz = torch.stack([
            xxx, yyy, zzz
        ], dim=-1).numpy()
        return xyz # x, y, z, 3

    def __call__(self, results):
        # input is the occupancy annotation file generated by SurroundOcc
        # results['pts_filename'] is the path to the point cloud file
        label_file = os.path.join(self.occ_path, results['pts_filename'].split('/')[-1]+'.npy')
        if os.path.exists(label_file):
            label = np.load(label_file)

            # (200, 200, 16) is the shape of the grid
            # 17 is the number of classes
            new_label = np.ones((200, 200, 16), dtype=np.int64) * 17
            # Give the new label the value of the original label
            new_label[label[:, 0], label[:, 1], label[:, 2]] = label[:, 3]

            # Define a mask to see which grid cells are occupied
            # From SurroundOcc github, 0 is ignored class which is set to be 255. Here we use a mask.
            # In the head, we set empty label to be 17 (no annotation), and the regression number classes is 18: 0, 1-16, 17
            mask = new_label != 0

            # Update results
            results['occ_label'] = new_label if self.semantic else new_label != 17
            results['occ_cam_mask'] = mask
        elif self.use_sweeps:
            new_label = np.ones((200, 200, 16), dtype=np.int64) * 17
            mask = new_label != 0
            results['occ_label'] = new_label if self.semantic else new_label != 17
            results['occ_cam_mask'] = mask
        else:
            raise NotImplementedError
        
        """
        Since SurroundOcc's annotation is in the camera coordinate system, we need to convert the ego frame to the lidar coordinate system if we are using ego!!
        """
        if not self.use_ego:
            occ_xyz = self.xyz[..., :3]
        else:
            ego2lidar = np.linalg.inv(results['ego2lidar']) # 4, 4
            occ_xyz = ego2lidar[None, None, None, ...] @ self.xyz[..., None] # x, y, z, 4, 1
            occ_xyz = np.squeeze(occ_xyz, -1)[..., :3]
        
        results['occ_xyz'] = occ_xyz
        
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class LoadOccupancyOcc3d(object):

    def __init__(
        self,
        occ_path,
        semantic=False,
        use_ego=False,
        use_occ3d_mask=False,
        pc_range=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
        visibility_pc_range=None,
        points_pc_range=None,
        use_lidar=True,
        use_mask_training=False,
        transform_gt_boxes=True,
        debug=False,
        debug_interval=50,
        debug_max_print=4,
        debug_rank0_worker0_only=True,
        **kwargs,
    ):
        self.occ_path = occ_path
        self.semantic = semantic
        self.use_ego = use_ego
        assert semantic and use_ego
        self.use_occ3d_mask = use_occ3d_mask

        # Occ3D labels/loss are always defined on the official 80m x 80m range.
        self.pc_range = [float(v) for v in pc_range]

        # The measured-Gaussian branch needs a wider support range.  Keep both
        # model points and 7D visibility_points in this wider range before the
        # later voxelizer/head applies the official Occ3D supervision range.
        self.visibility_pc_range = (
            [float(v) for v in visibility_pc_range]
            if visibility_pc_range is not None else list(self.pc_range)
        )
        self.points_pc_range = (
            [float(v) for v in points_pc_range]
            if points_pc_range is not None else list(self.visibility_pc_range)
        )

        # Occ3D labels and the point cloud branch are converted to ego frame
        # below when use_ego=True.  DAOcc/MMDet3D detection boxes are usually
        # stored in the LiDAR frame, so they must be converted by the same
        # lidar2ego transform before GT-box Gaussian completion consumes them.
        self.transform_gt_boxes = bool(transform_gt_boxes)
        self.use_lidar = use_lidar
        self.use_mask_training = use_mask_training

        # Optional local debug.  It is off by default; dataset-level coord debug
        # is normally enough.  When enabled, print only from rank0/worker0.
        self.debug = bool(debug)
        self.debug_interval = max(int(debug_interval), 1)
        self.debug_max_print = max(int(debug_max_print), 0)
        self.debug_rank0_worker0_only = bool(debug_rank0_worker0_only)
        self._debug_call_count = 0
        self._debug_print_count = 0
        if kwargs:
            self._ignored_kwargs = sorted(list(kwargs.keys()))
        else:
            self._ignored_kwargs = []

        xyz = self.get_meshgrid(self.pc_range, [200, 200, 16], 0.4)
        self.xyz = np.concatenate([xyz, np.ones_like(xyz[..., :1])], axis=-1) # x, y, z, 4

    def get_meshgrid(self, ranges, grid, reso):
        xxx = torch.arange(grid[0], dtype=torch.float) * reso + 0.5 * reso + ranges[0]
        yyy = torch.arange(grid[1], dtype=torch.float) * reso + 0.5 * reso + ranges[1]
        zzz = torch.arange(grid[2], dtype=torch.float) * reso + 0.5 * reso + ranges[2]

        xxx = xxx[:, None, None].expand(*grid)
        yyy = yyy[None, :, None].expand(*grid)
        zzz = zzz[None, None, :].expand(*grid)

        xyz = torch.stack([
            xxx, yyy, zzz
        ], dim=-1).numpy()
        return xyz # x, y, z, 3

    @staticmethod
    def _normalize_yaw_np(yaw):
        return (yaw + np.pi) % (2.0 * np.pi) - np.pi

    @staticmethod
    def _transform_boxes_lidar_to_ego_np(boxes, lidar2ego_rot, lidar2ego_tran):
        """Transform [x,y,z,l,w,h,yaw] or [x,y,z,w,l,h,yaw] boxes to ego.

        The dimension order is intentionally left untouched.  Only center xyz
        and yaw are transformed.  Yaw is treated with the same convention used
        by BoxGaussianCompletionHead: forward=(cos(yaw), sin(yaw), 0).
        """
        if boxes is None:
            return boxes
        boxes_np = np.asarray(boxes, dtype=np.float32)
        if boxes_np.ndim != 2 or boxes_np.shape[1] < 7 or boxes_np.shape[0] == 0:
            return boxes
        out = boxes_np.copy()
        centers = out[:, :3].astype(np.float64, copy=False)
        out[:, :3] = (centers @ lidar2ego_rot.T + lidar2ego_tran).astype(np.float32)

        yaw = out[:, 6].astype(np.float64, copy=False)
        direction_lidar = np.stack(
            [np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)],
            axis=-1,
        )
        direction_ego = direction_lidar @ lidar2ego_rot.T
        out[:, 6] = LoadOccupancyOcc3d._normalize_yaw_np(
            np.arctan2(direction_ego[:, 1], direction_ego[:, 0])
        ).astype(np.float32)
        return out

    def _maybe_transform_gt_boxes_to_ego(self, results, lidar2ego_rot, lidar2ego_tran):
        if not self.transform_gt_boxes or 'gt_bboxes_3d' not in results:
            return
        boxes = results.get('gt_bboxes_3d', None)
        if boxes is None:
            return
        try:
            results['gt_bboxes_3d'] = self._transform_boxes_lidar_to_ego_np(
                boxes, lidar2ego_rot, lidar2ego_tran
            )
            results['gt_bboxes_3d_coord'] = 'ego'
        except Exception:
            # Keep training robust; downstream debug can reveal malformed boxes.
            results['gt_bboxes_3d_coord'] = 'unknown_transform_failed'

    def _resolve_occ3d_label_file(self, results):
        """Resolve Occ3D labels.npz robustly for both GF3D and DAOcc pkl styles.

        Supported forms of results['occ_path'] include:
          - absolute path to labels.npz
          - absolute path to a directory containing labels.npz
          - relative path like gts/<scene>/<sample>/labels.npz
          - relative path like <scene>/<sample>/labels.npz
          - relative path to a directory containing labels.npz
        """
        occ_rel = results.get('occ_path', None)
        candidates = []

        def add(path):
            if path is None:
                return
            path = os.path.normpath(str(path))
            if path not in candidates:
                candidates.append(path)

        def add_file_or_dir(path):
            if path is None:
                return
            path = os.path.normpath(str(path))
            add(path)
            if not path.endswith('labels.npz'):
                add(os.path.join(path, 'labels.npz'))

        if occ_rel is not None:
            occ_rel = str(occ_rel)
            # As stored in info pkl.
            add_file_or_dir(occ_rel)

            # Under configured occ root, e.g. data/nuscenes/gts.
            if not os.path.isabs(occ_rel):
                add_file_or_dir(os.path.join(self.occ_path, occ_rel))

                # If occ_rel starts with gts/... and self.occ_path already ends with gts,
                # join with data/nuscenes instead of data/nuscenes/gts/gts/...
                occ_root_parent = os.path.dirname(os.path.normpath(self.occ_path))
                add_file_or_dir(os.path.join(occ_root_parent, occ_rel))

                # If occ_rel is data/nuscenes/gts/... but the current cwd is repo root,
                # the as-is candidate above already covers it. This branch covers ./ prefix.
                if occ_rel.startswith('./'):
                    add_file_or_dir(occ_rel[2:])

        # Fallback from scene/sample metadata.
        scene = results.get('scene_name', results.get('scene_token', None))
        sample = results.get('sample_idx', results.get('sample_token', None))
        if scene is not None and sample is not None:
            add_file_or_dir(os.path.join(self.occ_path, str(scene), str(sample)))
            add_file_or_dir(os.path.join(os.path.dirname(os.path.normpath(self.occ_path)), 'gts', str(scene), str(sample)))

        for cand in candidates:
            if os.path.exists(cand) and os.path.isfile(cand):
                return cand, candidates
        return None, candidates

    def __call__(self, results):
        label_file, candidates = self._resolve_occ3d_label_file(results)
        if label_file is None:
            raise FileNotFoundError(
                'Cannot find Occ3D labels.npz. Tried:\n  ' + '\n  '.join(candidates[:32])
            )

        occ3d = np.load(label_file)
        label = occ3d['semantics'] # (200, 200, 16)
        occ3d_mask_camera = occ3d['mask_camera'] # (200, 200, 16)
        occ3d_mask_lidar = occ3d['mask_lidar'] # (200, 200, 16)

        new_label = np.ones((200, 200, 16), dtype=np.int64) * 17 # 17 is empty
        new_label = label

        if self.use_occ3d_mask and self.use_mask_training:
            mask = occ3d_mask_camera
            results['occ3d_mask_camera'] = occ3d_mask_camera
        elif self.use_occ3d_mask and not self.use_mask_training:
            mask = np.ones((200, 200, 16), dtype=np.int64)
            results['occ3d_mask_camera'] = occ3d_mask_camera
        else:
            mask = np.ones((200, 200, 16), dtype=np.int64)

        results['occ_label'] = new_label
        results['occ_cam_mask'] = mask
        results['occ_gt_path'] = label_file

        # Occ3D is in ego frame, so convert LiDAR-frame point clouds to ego frame.
        if self.use_ego and self.use_lidar:
            lidar2ego = np.linalg.inv(results['ego2lidar'])
            lidar2ego_rot = lidar2ego[:3, :3]
            lidar2ego_tran = lidar2ego[:3, 3]

            points = results['points'].tensor[..., :3]
            points = points @ lidar2ego_rot.T + lidar2ego_tran
            results['points'].tensor[..., :3] = points

            if 'visibility_points' in results:
                visibility_points_xyz = results['visibility_points'].tensor[..., :3]
                visibility_points_xyz = visibility_points_xyz @ lidar2ego_rot.T + lidar2ego_tran
                results['visibility_points'].tensor[..., :3] = visibility_points_xyz

            # Keep GT boxes in the same ego/Occ3D frame as points,
            # visibility_points, occ_xyz, and projection_mat when use_ego=True.
            self._maybe_transform_gt_boxes_to_ego(results, lidar2ego_rot, lidar2ego_tran)

            new_points = results['points'].tensor
            p_range = self.points_pc_range
            new_points_mask = (new_points[..., 0] > p_range[0]) & (new_points[..., 0] < p_range[3]) & \
                              (new_points[..., 1] > p_range[1]) & (new_points[..., 1] < p_range[4]) & \
                              (new_points[..., 2] > p_range[2]) & (new_points[..., 2] < p_range[5])
            results['points'].tensor = new_points[new_points_mask]

            if 'visibility_points' in results:
                new_visibility_points = results['visibility_points'].tensor
                v_range = self.visibility_pc_range
                visibility_mask = (new_visibility_points[..., 0] > v_range[0]) & (new_visibility_points[..., 0] < v_range[3]) & \
                                  (new_visibility_points[..., 1] > v_range[1]) & (new_visibility_points[..., 1] < v_range[4]) & \
                                  (new_visibility_points[..., 2] > v_range[2]) & (new_visibility_points[..., 2] < v_range[5])
                results['visibility_points'].tensor = new_visibility_points[visibility_mask]

            self._maybe_print_debug(results)

        occ_xyz = self.xyz[..., :3]
        results['occ_xyz'] = occ_xyz
        return results

    @staticmethod
    def _range_str(points):
        try:
            pts = points.tensor if hasattr(points, 'tensor') else points
            if isinstance(pts, torch.Tensor):
                if pts.numel() == 0:
                    return 'empty'
                xyz = pts[..., :3].detach().float()
                return (
                    f"shape={tuple(pts.shape)}, "
                    f"x=[{float(xyz[:, 0].min()):.3f},{float(xyz[:, 0].max()):.3f}], "
                    f"y=[{float(xyz[:, 1].min()):.3f},{float(xyz[:, 1].max()):.3f}], "
                    f"z=[{float(xyz[:, 2].min()):.3f},{float(xyz[:, 2].max()):.3f}]"
                )
            arr = np.asarray(pts)
            if arr.size == 0:
                return 'empty'
            xyz = arr[..., :3].reshape(-1, 3)
            return (
                f"shape={arr.shape}, "
                f"x=[{float(np.nanmin(xyz[:, 0])):.3f},{float(np.nanmax(xyz[:, 0])):.3f}], "
                f"y=[{float(np.nanmin(xyz[:, 1])):.3f},{float(np.nanmax(xyz[:, 1])):.3f}], "
                f"z=[{float(np.nanmin(xyz[:, 2])):.3f},{float(np.nanmax(xyz[:, 2])):.3f}]"
            )
        except Exception as exc:
            return f'range_failed({exc})'

    @staticmethod
    def _is_main_process():
        rank = os.environ.get('RANK') or os.environ.get('LOCAL_RANK')
        return rank in (None, '', '0')

    @staticmethod
    def _is_worker_zero():
        try:
            from torch.utils.data import get_worker_info
            worker = get_worker_info()
            return worker is None or int(worker.id) == 0
        except Exception:
            return True

    def _maybe_print_debug(self, results):
        if not self.debug:
            return
        self._debug_call_count += 1
        if self._debug_print_count >= self.debug_max_print:
            return
        if self._debug_call_count % self.debug_interval != 0:
            return
        if self.debug_rank0_worker0_only and (not self._is_main_process() or not self._is_worker_zero()):
            return
        msg = [
            f"[LoadOccupancyOcc3d RANGE DEBUG] call={self._debug_call_count};",
            f"pc_range={self.pc_range}; points_pc_range={self.points_pc_range}; visibility_pc_range={self.visibility_pc_range};",
        ]
        if 'points' in results:
            msg.append('points=' + self._range_str(results['points']))
        if 'visibility_points' in results:
            msg.append('visibility_points=' + self._range_str(results['visibility_points']))
        if self._ignored_kwargs:
            msg.append(f"ignored_kwargs={self._ignored_kwargs};")
        print(' '.join(msg), flush=True)
        self._debug_print_count += 1

    def __repr__(self):
        """str: Return a string that describes the module."""
        return (
            f"{self.__class__.__name__}(pc_range={self.pc_range}, "
            f"points_pc_range={self.points_pc_range}, "
            f"visibility_pc_range={self.visibility_pc_range})"
        )


@OPENOCC_TRANSFORMS.register_module()
class LoadOccupancyOcc3DCompat(LoadOccupancyOcc3d):
    """Compatibility alias used by Occ3D configs.

    The implementation is exactly LoadOccupancyOcc3d: it loads Occ3D labels,
    converts LiDAR-frame points to ego frame when use_ego=True, and applies the
    Occ3D labels/6D encoder points use pc_range; 7D visibility_points may use
    visibility_pc_range so ground/long-range octree support is not cut by the
    official label range.
    """
    pass


@OPENOCC_TRANSFORMS.register_module()
class LoadPointFromFileLiDARWildOcc(object):
    """Load LiDAR Points From File.

    Load sunrgbd and scannet points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
    """

    def __init__(
        self,
        coord_type='LIDAR',
        load_dim=4, # for our case: 4
        use_dim=[0, 1, 2, 3], # for our case: 4
        shift_height=False,
        use_color=False,
        load_augmented=None,
        reduce_beams=None,
    ):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert (
            max(use_dim) < load_dim
        ), f"Expect all used dimensions < {load_dim}, got {use_dim}"
        assert coord_type in ["CAMERA", "LIDAR", "DEPTH"]

        self.coord_type = coord_type
        self.load_dim = load_dim
        assert load_dim == 5
        self.use_dim = use_dim
        self.load_augmented = load_augmented
        self.reduce_beams = reduce_beams

    def _load_points(self, lidar_path):
        """Private function to load point clouds data.

        Args:
            lidar_path (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        # mmcv.check_file_exist(lidar_path)
        mmengine.check_file_exist(lidar_path)
        if lidar_path.endswith(".npy"):
            points = np.load(lidar_path)
        else:
            points = np.fromfile(lidar_path, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        lidar_path = results["lidar_path"]
        # lidar_path = results["pts_filename"]
        points = self._load_points(lidar_path)
        points = points.reshape(-1, self.load_dim-1)
        if points.shape[1] == 4:
            points = np.pad(points, ((0, 0), (0, 1)), mode='constant', constant_values=0.0)
        points[:, 4] = 0
        # check reduced beams
        if self.reduce_beams and self.reduce_beams < 32:
            points = reduce_LiDAR_beams(points, self.reduce_beams)
        points = points[:, self.use_dim]
        attribute_dims = None

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims
        )
        results["points"] = points

        return results
    
    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class LoadPointsFromMultiSweepsLiDARWildOcc(object):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(
        self,
        sweeps_num=10,
        load_dim=5,
        use_dim=[0, 1, 2, 4],
        pad_empty_sweeps=False,
        remove_close=False,
        test_mode=False,
        load_augmented=None,
        reduce_beams=None,
    ):
        self.load_dim = load_dim
        assert load_dim == 5
        self.sweeps_num = sweeps_num
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        self.use_dim = use_dim
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        self.load_augmented = load_augmented
        self.reduce_beams = reduce_beams

    def _load_points(self, lidar_path):
        """Private function to load point clouds data.

        Args:
            lidar_path (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        # mmcv.check_file_exist(lidar_path)
        mmengine.check_file_exist(lidar_path)
        if self.load_augmented:
            assert self.load_augmented in ["pointpainting", "mvp"]
            virtual = self.load_augmented == "mvp"
            points = load_augmented_point_cloud(
                lidar_path, virtual=virtual, reduce_beams=self.reduce_beams
            )
        elif lidar_path.endswith(".npy"):
            points = np.load(lidar_path)
        else:
            points = np.fromfile(lidar_path, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        points = results["points"]
        if points.shape[1] == 4:
            points = np.pad(points, ((0, 0), (0, 1)), mode='constant', constant_values=0.0)
        points = points[:, self.use_dim]
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results["timestamp"] / 1e3 # swei: convert to ms, original is 1e6
        if self.pad_empty_sweeps and len(results["sweeps"]) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results["sweeps"]) <= self.sweeps_num:
                choices = np.arange(len(results["sweeps"]))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                # NOTE: seems possible to load frame -11?
                if not self.load_augmented:
                    choices = np.random.choice(
                        len(results["sweeps"]), self.sweeps_num, replace=False
                    )
                else:
                    # don't allow to sample the earliest frame, match with Tianwei's implementation.
                    choices = np.random.choice(
                        len(results["sweeps"]) - 1, self.sweeps_num, replace=False
                    )
            for idx in choices:
                sweep = results["sweeps"][idx]
                points_sweep = self._load_points(sweep["lidar_path"])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim-1)
                if points_sweep.shape[1] == 4:
                    points_sweep = np.pad(points_sweep, ((0, 0), (0, 1)), mode='constant', constant_values=0.0)

                if self.reduce_beams and self.reduce_beams < 32:
                    points_sweep = reduce_LiDAR_beams(points_sweep, self.reduce_beams)

                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                points_sweep = points_sweep[:, self.use_dim] # [num_points, 4]
                sweep_ts = sweep["timestamp"] / 1e3 # convert to ms, original is 1e6
                # TODO: swei: check whether this is correct
                sweep2lidar = results['lidar_pose'] @ np.linalg.inv(sweep['lidar_pose'])
                sweep2lidar_rotation = sweep2lidar[:3, :3]
                sweep2lidar_translation = sweep2lidar[:3, 3]
                points_sweep[:, :3] = (
                    points_sweep[:, :3] @ sweep2lidar_rotation
                )
                points_sweep[:, :3] += sweep2lidar_translation
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)

        results["points"] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f"{self.__class__.__name__}(sweeps_num={self.sweeps_num})"


@OPENOCC_TRANSFORMS.register_module()
class LoadImageFromFilesWildOcc(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool, optional): Whether to convert the img to float32.
            Defaults to False.
        color_type (str, optional): Color type of the file.
            Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['image_path']
        # img is of shape (h, w, c)
        img = mmcv.imread(filename, self.color_type)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        results['img'] = [img] # single-view image
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class LoadOccupancyWildOcc(object):
    def __init__(self):
        # 3: 'grass', 4: 'tree', 19: 'bush', 31: 'puddle', 33: 'mud', 27: 'barrier', 34: 'rubble'
        # reference: https://github.com/unmannedlab/RELLIS-3D/blob/main/utils/label2color.ipynb
        self.label_wildocc = [3, 4, 19, 31, 33, 27, 34]

        xyz = self.get_meshgrid([-20, -10, -2.0, 0, 10, 6.0], [100, 100, 40], 0.2)
        self.xyz = np.concatenate([xyz, np.ones_like(xyz[..., :1])], axis=-1) # x, y, z, 4

    def get_meshgrid(self, ranges, grid, reso):
        # NOTE: Attention: the index of the grid is the start of the FOV of the camera
        '''
                     |
                     |------o (start the index of the grid)
                     |      |
                   \ | /    |  (\ / is the FOV of the camera)
        y <----------o----------------
                     |
                     |
                     v
                     x
        '''
        xxx = torch.arange(grid[0], dtype=torch.float) * reso + 0.5 * reso + ranges[0]
        yyy = torch.arange(grid[1], dtype=torch.float) * reso + 0.5 * reso + ranges[1]
        zzz = torch.arange(grid[2], dtype=torch.float) * reso + 0.5 * reso + ranges[2]

        xxx = xxx[:, None, None].expand(*grid)
        yyy = yyy[None, :, None].expand(*grid)
        zzz = zzz[None, None, :].expand(*grid)

        xyz = torch.stack([
            xxx, yyy, zzz
        ], dim=-1).numpy()
        return xyz # x, y, z, 3

    def __call__(self, results):
        # input is the occupancy annotation file generated by SurroundOcc
        # results['pts_filename'] is the path to the point cloud file
        label_file = results['occ_path']
        label = np.load(label_file)
        # The shape of each npy file is (n,4), where n is the number of non-empty occupancies. Four dimensions represent xyz and semantic label respectively.

        # 17 is the number of classes
        new_label = np.ones((100, 100, 40), dtype=np.int64) * 8
        # Give the new label the value of the original label
        for idx, label_id in enumerate(self.label_wildocc):
            indices = label[label[:, 3] == label_id][:, :3]  # (N, 3)
            new_label[indices[:, 0], indices[:, 1], indices[:, 2]] = idx + 1 
        indices = label[~np.isin(label[:, 3], self.label_wildocc)][:, :3]
        new_label[indices[:, 0], indices[:, 1], indices[:, 2]] = 0

        # Define a mask to see which grid cells are occupied
        # From SurroundOcc github, 0 is ignored class which is set to be 255. Here we use a mask.
        # In the head, we set empty label to be 17 (no annotation), and the regression number classes is 9: 0, 1-7, 17
        mask = new_label != 0

        # Update results
        results['occ_label'] = new_label
        results['occ_cam_mask'] = mask
        
        """
        Since SurroundOcc's annotation is in the camera coordinate system, we need to convert the ego frame to the lidar coordinate system if we are using ego!!
        """
        occ_xyz = self.xyz[..., :3]
        
        results['occ_xyz'] = occ_xyz
        
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@OPENOCC_TRANSFORMS.register_module()
class LoadDepthFromFileWildOcc(object):
    """Load the gound truth depth map generated from BEVDepth using lidar.
    """

    def __init__(self, is_to_depth_map=True, map_size=None):
        self.is_to_depth_map = is_to_depth_map
        self.map_size = map_size

    def __call__(self, results):
        if self.map_size is None:
            self.map_size = results['img'][0].shape[:2] # List of (H, W, C) numpy array, here should be (1200, 1920)
        img_paths = results['img_filename']
        dpt_paths = []
        map_depths = []

        dpt_path = results['depth_path']
        point_depth = np.fromfile(dpt_path, dtype=np.float32, count=-1).reshape(-1, 3)
        dpt_paths.append(dpt_path)
        if self.is_to_depth_map:
            map_depth = self.to_depth_map(point_depth)
            map_depths.append(map_depth)
        
        # img is of shape (h, w, c, num_views)
        # map_depths is a List of depth maps, each of shape (H, W)
        # N * (H, W), N = num_views
        results['dpt'] = map_depths
        results['filename_dpt'] = dpt_paths
        return results
    
    def to_depth_map(self, point_depth):
        """Transform depth based on ida augmentation configuration.

        Args:
            cam_depth (np array): Nx3, 3: x,y,d.
            resize (float): Resize factor.
            resize_dims (list): Final dimension.
            crop (list): x1, y1, x2, y2
            flip (bool): Whether to flip.
            rotate (float): Rotation value.

        Returns:
            np array: [h/down_ratio, w/down_ratio, d]
        """

        # Here they assume the point depth coordinates are 900, 1600?
        # TODO: check the point depth coordinate is (H, W) or (W, H)
        depth_coords = point_depth[:, :2].astype(np.int16)

        # The loaded image shape is also 900, 1600?
        depth_map = np.zeros(self.map_size) # (H, W)
        valid_mask = ((depth_coords[:, 1] < self.map_size[0])
                    & (depth_coords[:, 0] < self.map_size[1])
                    & (depth_coords[:, 1] >= 0)
                    & (depth_coords[:, 0] >= 0))
        depth_map[depth_coords[valid_mask, 1],
                depth_coords[valid_mask, 0]] = point_depth[valid_mask, 2]

        return depth_map
    
    def __repr__(self):
        return self.__class__.__name__
