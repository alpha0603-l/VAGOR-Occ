# ================== data: nuScenes-Occ3D for GaussianFormer3D ========================
# Put nuScenes raw data at data/nuscenes via symlink.
# Put Occ3D GT so that labels.npz can be found under one of these forms:
#   data/nuscenes/gts/<scene>/<sample_token>/labels.npz
#   data/nuscenes/Occ3D/gts/<scene>/<sample_token>/labels.npz
# or keep info['occ_path'] in the pkl and set occ_path to its parent root.

data_root = "data/nuscenes/"
anno_root = "data/nuscenes_cam/"
occ_path = "data/nuscenes/gts"
input_shape = (704, 256)
batch_size = 1

# 50-iter lightweight coordinate/label debug.  Keep this on while fixing
# Occ3D/DAOcc coordinate and label compatibility; capped by max_print.
dataset_coord_debug = dict(
    enabled=True,
    interval=50,
    max_print=12,
    show_matrices=False,
    show_label_hist=True,
)
# Occ3D-nuScenes uses [-40, -40, -1, 40, 40, 5.4], voxel grid [200, 200, 16].
occ_pc_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
# DAOcc measured-Gaussian branch keeps the wider/lower LiDAR support range.
visibility_pc_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 5.4]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

# NOTE:
# Depth loading is enabled for Sparse-UVD GAF.  The depth_gt/*.bin files are LiDAR-projected
# sparse depth maps used to supervise the image depth distribution.
train_pipeline = [
    dict(type="LoadPointFromFileLiDAR", coord_type="LIDAR", load_dim=5, use_dim=5),
    dict(
        type="LoadPointsFromMultiSweepsLiDAR",
        sweeps_num=10,
        load_dim=5,
        use_dim=6,                 # LiDAR encoder: x,y,z,intensity,time_lag,decay
        visibility_use_dim=7,      # Visibility/octree: x,y,z,intensity,time_lag,decay,is_current
        pad_empty_sweeps=True,
        remove_close=True,
        add_decay=True,
        decay_lambda=1.0,
        decay_normalize=True,
        add_is_current=True,
        point_cloud_range=visibility_pc_range,
        crop_points=True,
        sort_sweeps_by_time=True,
        debug=False,
    ),
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadMultiViewDepthFromFiles", is_to_depth_map=True, map_size=None),
    # Actual image resize/crop to 704x256.  Dataset._sample_augmentation
    # produces aug_configs from data_aug_conf; ResizeCropFlipImage updates
    # ego2img/lidar2img, so GAF projection sampling stays aligned.
    dict(type="ResizeCropFlipImage"),
    dict(
        type="LoadOccupancyOcc3d",
        occ_path=occ_path,
        semantic=True,
        use_ego=True,
        use_occ3d_mask=True,
        pc_range=occ_pc_range,
        # Keep point clouds/visibility support wider than the Occ3D label grid.
        # The final occupancy target/head still uses occ_pc_range; this wider range
        # only feeds measured Gaussian generation and boundary free-space context.
        visibility_pc_range=visibility_pc_range,
        points_pc_range=visibility_pc_range,
        use_lidar=True,
        use_mask_training=True,
    ),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="NuScenesAdaptor", use_ego=True, num_cams=6),
]

test_pipeline = [
    dict(type="LoadPointFromFileLiDAR", coord_type="LIDAR", load_dim=5, use_dim=5),
    dict(
        type="LoadPointsFromMultiSweepsLiDAR",
        sweeps_num=10,
        load_dim=5,
        use_dim=6,                 # LiDAR encoder: x,y,z,intensity,time_lag,decay
        visibility_use_dim=7,      # Visibility/octree: x,y,z,intensity,time_lag,decay,is_current
        pad_empty_sweeps=True,
        remove_close=True,
        add_decay=True,
        decay_lambda=1.0,
        decay_normalize=True,
        add_is_current=True,
        point_cloud_range=visibility_pc_range,
        crop_points=True,
        sort_sweeps_by_time=True,
        debug=False,
    ),
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadMultiViewDepthFromFiles", is_to_depth_map=True, map_size=None),
    # Actual image resize/crop to 704x256.  Dataset._sample_augmentation
    # produces aug_configs from data_aug_conf; ResizeCropFlipImage updates
    # ego2img/lidar2img, so GAF projection sampling stays aligned.
    dict(type="ResizeCropFlipImage"),
    dict(
        type="LoadOccupancyOcc3d",
        occ_path=occ_path,
        semantic=True,
        use_ego=True,
        use_occ3d_mask=True,
        pc_range=occ_pc_range,
        # Keep point clouds/visibility support wider than the Occ3D label grid.
        # The final occupancy target/head still uses occ_pc_range; this wider range
        # only feeds measured Gaussian generation and boundary free-space context.
        visibility_pc_range=visibility_pc_range,
        points_pc_range=visibility_pc_range,
        use_lidar=True,
        use_mask_training=True,
    ),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="NuScenesAdaptor", use_ego=True, num_cams=6),
]

# Resize raw nuScenes 1600x900 images by a fixed 0.44 scale to 704x396,
# then crop to final 704x256.  This keeps aspect ratio and avoids hard warping.
data_aug_conf = dict(
    H=900,
    W=1600,
    final_dim=(256, 704),
    resize_lim=(0.44, 0.44),
    bot_pct_lim=(0.0, 0.0),
    rot_lim=(0.0, 0.0),
    rand_flip=False,
)

# Preferred pkl names. If your DAOcc generated files are named differently, either symlink them
# to these names or edit imageset below.
train_dataset_config = dict(
    type="NuScenesOcc3DDataset",
    data_root=data_root,
    imageset=anno_root + "nuscenes_infos_gf3d_occ3d_train.pkl",
    data_aug_conf=data_aug_conf,
    pipeline=train_pipeline,
    phase="train",
    respect_valid_flag=False,
    coord_debug=dataset_coord_debug,
    sample_interval=14,
    sample_offset=0,
)

val_dataset_config = dict(
    type="NuScenesOcc3DDataset",
    data_root=data_root,
    imageset=anno_root + "nuscenes_infos_gf3d_occ3d_val.pkl",
    data_aug_conf=data_aug_conf,
    pipeline=test_pipeline,
    phase="val",
    respect_valid_flag=False,
    coord_debug=dict(enabled=True, interval=50, max_print=6, show_matrices=False, show_label_hist=True),
    sample_interval=1,
    sample_offset=0,
)

train_loader = dict(
    batch_size=batch_size,
    num_workers=2,
    shuffle=True,
)

val_loader = dict(
    batch_size=batch_size,
    num_workers=2,
)
