# -*- coding: utf-8 -*-
"""
Script for volume registration and cell mapping using ClearMap2.
Organized for clarity and maintainability.
"""
#make sure input points are in pixel coordinates of the stitched image
#if spatial coordinate, transform_points(indices=False)
#nohup python cellMap_fengyi2.py &> /data/hdd12tb-1/fengyi/COMBINe/clearmap_p5_trimmed_atlas/log/.txt &
#conda activate ClearMap
#change annotation file name in /ClearMap/Alignment/Annotation.py
import os
import csv
import shutil
import numpy as np
from ClearMap.Environment import *
import numpy.lib.recfunctions as rfn

# ==== === Project/Experiment Parameters === ===
STITCHED_FILENAME = 'registration.tif'
DATA_DIR = '/data/hdd12tb-1/fengyi/COMBINe/clearmap_p5_trimmed_atlas/fw9_'
CELL_CENTROIDS_DIR = os.path.join(DATA_DIR, 'cell_centroids')
RESOURCES_DIR = None # will be set later by settings.resources_path

#resolution: all pixel coordinates
VOXEL_SIZE_ORIGINAL = np.array([0.65, 0.65, 20])     # raw data resolution
VOXEL_SIZE_STITCHED = np.array([5.2,5.2, 10])        # pixel size of registration.tif: compare the max resolution stitched images (0.65um pixel size) to the unknown stitched image 
                                                     # normally the 5th resolution in terastitcher is 5.2um pixel size 
                                                     # z voxel size could be different from original size due to stitching
VOXEL_SIZE_RESAMPLED = np.array([20, 20, 20])        # resampled/atlas
N_CLASSES = 6                                        # number of cell classes

ratio = VOXEL_SIZE_STITCHED / VOXEL_SIZE_ORIGINAL

# ==== ==== Workspace Setup ==== ====
ws = wsp.Workspace('CellMap', directory=DATA_DIR)
ws.update(stitched=STITCHED_FILENAME)
ws.debug = False
RESOURCES_DIR = settings.resources_path
ws.info()

# ==== ==== Workspace Setup ==== ====
ws = wsp.Workspace('CellMap', directory=DATA_DIR)
ws.update(stitched=STITCHED_FILENAME)
ws.debug = False
RESOURCES_DIR = settings.resources_path
ws.info()

# --- 新增：打印自定义参数到日志 ---
import sys

print("\n" + "="*30)
print("EXPERIMENT PARAMETERS LOG")
print("="*30)
print(f"Data Directory:         {DATA_DIR}")
print(f"Stitched Filename:      {STITCHED_FILENAME}")
print(f"Cell Centroids Dir:     {CELL_CENTROIDS_DIR}")
print(f"Number of Cell Classes: {N_CLASSES}")
print("-" * 30)
print(f"Voxel Size Original:    {VOXEL_SIZE_ORIGINAL}")
print(f"Voxel Size Stitched:    {VOXEL_SIZE_STITCHED}")
print(f"Voxel Size Resampled:   {VOXEL_SIZE_RESAMPLED}")
print(f"Calculated Ratio:       {ratio}")
print("="*30 + "\n")

# 强制刷新缓冲区，确保 nohup 性能实时查看到以上内容
sys.stdout.flush()

# ==== ==== Annotation & Reference Preparation ==== ====
#output: ClearMap/Resources/Atlas/
def prepare_annotation():
    """Adjust and prepare annotation/reference files."""
    annotation_file, vol_annotation_file, reference_file = ano.prepare_annotation_files(
        slicing=(slice(None),slice(None),slice(None)), orientation=(1,-2,-3),   #change the order of axes ((-3,-1,-2) for 2017 adult atlas )
                                                                                #1,2,3: x,y,z
        overwrite=True, verbose=True)
    return annotation_file, vol_annotation_file, reference_file

(annotation_file, vol_annotation_file, reference_file) = prepare_annotation()

# Alignment parameter files
ALIGNMENT_PATH = os.path.join(RESOURCES_DIR, 'Alignment')
align_channels_affine_file   = io.join(ALIGNMENT_PATH, 'align_affine.txt')
align_reference_affine_file  = io.join(ALIGNMENT_PATH, 'align_affine.txt')
align_reference_bspline_file = io.join(ALIGNMENT_PATH, 'align_bspline.txt')

# ==== ==== Resampling ==== ==== 
#resample stitched image to atlas resolution
def resample_stitched():
    resample_parameter = {
        "source_resolution": tuple(VOXEL_SIZE_STITCHED),
        "sink_resolution": tuple(VOXEL_SIZE_RESAMPLED), #target resolution
        "processes": None,
        "verbose": True,
    }
    res.resample(ws.filename('stitched'), sink=ws.filename('resampled'), **resample_parameter)

resample_stitched()

# ==== ==== Alignment ==== ====
#check result.mhd in ws.filename('auto_to_reference') folder
def align_to_reference(): # Align resampled image to reference, save transform params
    align_reference_parameter = {
        "moving_image": reference_file, #moving the reference to the sample
        "fixed_image": ws.filename('resampled'),
        "affine_parameter_file": align_reference_affine_file,
        "bspline_parameter_file": align_reference_bspline_file,
        "result_directory": ws.filename('auto_to_reference')
    }
    elx.align(**align_reference_parameter)

align_to_reference()

# ==== ==== Cell Points Transformation & Annotation ==== ====
def transformation(coordinates):
    """Resample & transform coordinates to reference space."""
    coordinates = res.resample_points( 
        coordinates, sink=None, orientation=None, 
        source_shape=io.shape(ws.filename('stitched')),
        sink_shape=io.shape(ws.filename('resampled')) #downsample
    )
    coordinates = elx.transform_points(
        coordinates, sink=None,
        transform_directory=ws.filename('auto_to_reference'), 
        binary=False, 
        indices=True #indices = true for voxel coordinates, false for spatial coordinates
    )
    return coordinates

def insertdir(parent_file, i, name='cell_registration'):
    """Insert a label directory in file path."""
    dir_inserted = os.path.join(os.path.split(parent_file)[0], name, f'{i}')
    if not os.path.exists(dir_inserted):
        os.makedirs(dir_inserted)
    return os.path.join(dir_inserted, os.path.basename(parent_file))

def process_cell_class(class_idx):
    # 1. Load points for this class
    cell_points_file = os.path.join(CELL_CENTROIDS_DIR, f'ob_{class_idx}.csv')
    if not os.path.exists(cell_points_file):
        print(f"Skipping class {class_idx}: File not found.")
        return

    with open(cell_points_file, newline='') as csvfile:
        points = np.array(list(csv.reader(csvfile)), dtype=float)
    
    # 2. Transform coordinates
    coordinates = points / ratio  # downsampled to stitched image space
    coordinates_transformed = transformation(coordinates) #transformed to atlas space
    
    # 3. Annotation (使用内存中的 atlas_volume 数组) =========================
    
    # 3.1 转换为整数索引
    indices = np.round(coordinates_transformed).astype(int)
    
    # 3.2 边界检查 (使用 atlas_volume.shape)
    # 注意：这里我们使用全局变量 atlas_volume
    sh = atlas_volume.shape 
    
    # 确保索引不超出地图边界 (0 <= index < shape)
    # ClearMap 通常是 XYZ 顺序，这里假设 io.read 返回的也是 XYZ 顺序
    # 如果后续发现 label 错乱，可能需要改为 ZYX 顺序: indices[:, [2,1,0]]
    valid_x = (indices[:, 0] >= 0) & (indices[:, 0] < sh[0])
    valid_y = (indices[:, 1] >= 0) & (indices[:, 1] < sh[1])
    valid_z = (indices[:, 2] >= 0) & (indices[:, 2] < sh[2])
    valid_mask = valid_x & valid_y & valid_z
    
    # 3.3 直接从数组读取 ID
    raw_ids = np.zeros(len(indices), dtype=int) # 默认为 0
    
    if np.any(valid_mask):
        # 使用 Numpy 高级索引直接取值
        # 如果 io.read 保持了 XYZ 顺序：
        raw_ids[valid_mask] = atlas_volume[
            indices[valid_mask, 0],
            indices[valid_mask, 1],
            indices[valid_mask, 2]
        ]
    
    # --- 自定义逻辑 (处理 0 和 -1) ---
    label_values = np.zeros(len(raw_ids), dtype=int)
    name_values = np.array(['background'] * len(raw_ids), dtype='object')
    
    mask_0 = (raw_ids == 0)
    mask_neg1 = (raw_ids == -1)
    mask_valid_id = ~(mask_0 | mask_neg1)
    
    # 处理 -1
    if np.any(mask_neg1):
        name_values[mask_neg1] = 'no label'
        
    # 处理合法 ID (查表)
    if np.any(mask_valid_id):
        try:
            valid_ids = raw_ids[mask_valid_id]
            # 这里依然调用 ano.convert_label 查字典，但只查合法的 ID
            label_values[mask_valid_id] = ano.convert_label(valid_ids, key='id', value='graph_order')
            name_values[mask_valid_id] = ano.convert_label(valid_ids, key='id', value='name')
        except Exception as e:
            print(f"Warning: Mapping error in class {class_idx}. Details: {e}")
    
    # (结束自定义逻辑) =======================================================

    # 4. Voxelization
    voxelization_parameter = dict(
        shape=sh,  # 直接用 shape
        dtype=None,
        weights=None,
        method='sphere',
        radius=(1,1,1),
        kernel=None,
        processes=None,
        verbose=True
    )
    
    vox.voxelize(
        coordinates_transformed,
        sink=insertdir(ws.filename('density', postfix='counts'), class_idx),
        **voxelization_parameter
    )
    
    # 5. Save results
    # points.dtype = [(c, float) for c in ('x', 'y', 'z')]
    # coordinates_transformed.dtype = [(t, float) for t in ('xt', 'yt', 'zt')]
    points = points.view([(c, float) for c in ('x', 'y', 'z')])
    coordinates_transformed = coordinates_transformed.view([(t, float) for t in ('xt', 'yt', 'zt')])
    
    label_struct = np.array(label_values, dtype=[('graph_order', int)])
    names_struct = np.array(name_values, dtype=[('name', 'S256')])

    cells_data = rfn.merge_arrays([points, coordinates_transformed, label_struct, names_struct], flatten=True, usemask=False)
    
    io.write(insertdir(ws.filename('cell_registration'), class_idx), cells_data)
    np.savetxt(
        insertdir(ws.filename('cell_registration', extension='csv'), class_idx), 
        cells_data, delimiter=',', fmt='%s'
    )

print(f"Loading annotation volume from: {annotation_file}")
atlas_volume = io.read(annotation_file)
print(f"Atlas loaded. Shape: {atlas_volume.shape}")
print('Starting cell alignment...')
for i in range(N_CLASSES):
    process_cell_class(i)

# ==== ==== Transform Annotation Volume ==== ====
def transform_annotation_volume():
    path = settings.elastix_path
    transformix_binary = os.path.join(path, 'bin/transformix')
    vol_dir = os.path.join(DATA_DIR, 'volume')
    if not os.path.exists(vol_dir):
        os.makedirs(vol_dir)
    # copy transform parameters
    for i in [0, 1]:
        src = os.path.join(ws.filename('auto_to_reference'), f'TransformParameters.{i}.txt')
        shutil.copy2(src, vol_dir)
    transform_parameter_file = os.path.join(vol_dir, 'TransformParameters.1.txt')
    # set interpolation order = 0 for label data
    with open(transform_parameter_file, 'r') as file:
        filedata = file.read()
    filedata = filedata.replace('FinalBSplineInterpolationOrder 3', 'FinalBSplineInterpolationOrder 0')
    with open(transform_parameter_file, 'w') as file:
        file.write(filedata)
    # apply transform
    cmd = '{} -in {} -out {} -tp {}'.format(
        transformix_binary, vol_annotation_file, vol_dir, transform_parameter_file)
    os.system(cmd)

transform_annotation_volume()

