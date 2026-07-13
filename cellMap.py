# -*- coding: utf-8 -*-
"""
Script for volume registration and cell mapping using ClearMap2.
Organized for clarity and maintainability.
"""
#make sure input points are in pixel coordinates of the stitched image
#if spatial coordinate, transform_points(indices=False)
# nohup python cellMap.py &> /data/hdd12tb-1/fengyi/COMBINe/clearmap/TSC/s18/log.txt
import os
# Headless servers have no X11 display, but ClearMap.Environment pulls in
# Qt-based plotting modules on import (unused here); force the offscreen
# Qt backend so that import doesn't crash. Doesn't override if already set.
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import csv
import shutil
import numpy as np
import tifffile
from ClearMap.Environment import *
import numpy.lib.recfunctions as rfn
import glob
import yaml
import sys

# ==== 1. 加载配置文件 ====
def load_config(config_path='/home/fyu7/My_project/COMBINe/ClearMap/config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

cfg = load_config()

# ==== 2. 映射参数 ====
DATA_DIR = cfg['paths']['data_dir']

def ensure_stitched_npy(stitched_filename):
    """If enabled in config, convert stitched .tif -> .npy since ClearMap
    can't read very large TIF stacks directly. Skips if the .npy already
    exists."""
    if not cfg['paths'].get('convert_tif_to_npy', False):
        return stitched_filename

    base, _ = os.path.splitext(stitched_filename)
    npy_filename = base + '.npy'
    npy_path = os.path.join(DATA_DIR, npy_filename)
    tif_path = os.path.join(DATA_DIR, base + '.tif')

    if os.path.exists(npy_path):
        print(f"Found existing {npy_path}, skipping TIF->NPY conversion.")
        return npy_filename

    if not os.path.exists(tif_path):
        raise FileNotFoundError(
            f"convert_tif_to_npy is enabled but no TIF found at {tif_path}")

    print(f"Converting {tif_path} -> {npy_path} ...")
    img_array = tifffile.imread(tif_path)
    img_array = np.transpose(img_array, (2, 1, 0))  # (Z,Y,X) -> (X,Y,Z)
    tmp_path = npy_path + '.tmp.npy'
    np.save(tmp_path, img_array)
    os.replace(tmp_path, npy_path)
    print(f"Conversion complete: shape {img_array.shape}")
    return npy_filename

STITCHED_FILENAME = ensure_stitched_npy(cfg['paths']['stitched_filename'])
CELL_CENTROIDS_DIR = os.path.join(DATA_DIR, 'cell_centroids')

VOXEL_SIZE_ORIGINAL = np.array(cfg['resolution']['original'])
VOXEL_SIZE_STITCHED = np.array(cfg['resolution']['stitched'])
VOXEL_SIZE_RESAMPLED = np.array(cfg['resolution']['resampled'])
ratio = VOXEL_SIZE_STITCHED / VOXEL_SIZE_ORIGINAL
CELL_PREFIX = cfg.get('cells', {}).get('prefix', 'ob_')
SAVE_DENSITY_TIF = cfg.get('cells', {}).get('save_density_tif', True)

# 处理 Slicing 元组转换
def parse_slicing(slicing_list):
    if slicing_list is None: return None
    slices = []
    for s in slicing_list:
        if s is None:
            slices.append(slice(None))
        else:
            slices.append(slice(s[0], s[1]))
    return tuple(slices)

MY_SLICING = parse_slicing(cfg['registration']['slicing'])
MY_ORIENTATION = tuple(cfg['registration']['orientation'])
CROP_CONFIG = cfg['registration'].get('crop_for_registration', None)
CROP_OFFSET = np.array([0, 0, 0], dtype=float)  

# ==== ==== Workspace Setup ==== ====
ws = wsp.Workspace('CellMap', directory=DATA_DIR)
ws.update(stitched=STITCHED_FILENAME)
ws.debug = False
RESOURCES_DIR = settings.resources_path
ws.info()

# --- 打印自定义参数到日志 ---
print("\n" + "="*30)
print("EXPERIMENT PARAMETERS LOG")
print("="*30)
print(f"Data Directory:         {DATA_DIR}")
print(f"Stitched Filename:      {STITCHED_FILENAME}")
print(f"Cell Centroids Dir:     {CELL_CENTROIDS_DIR}")
print("-" * 30)
print(f"Voxel Size Original:    {VOXEL_SIZE_ORIGINAL}")
print(f"Voxel Size Stitched:    {VOXEL_SIZE_STITCHED}")
print(f"Voxel Size Resampled:   {VOXEL_SIZE_RESAMPLED}")
print(f"Calculated Ratio:       {ratio}")
print(f"Orientation:            {MY_ORIENTATION}")
print(f"Slicing:                {MY_SLICING}")
print("="*30 + "\n")

# 强制刷新缓冲区，确保 nohup 性能实时查看到以上内容
sys.stdout.flush()

# ==== ==== Annotation & Reference Preparation ==== ====
#output: ClearMap/Resources/Atlas/
def build_annotation_postfix(orientation, slicing):
    """Short, deterministic postfix derived from orientation/slicing (instead
    of ClearMap's default, which spells out the full slice-object repr and
    gets unwieldy long). Same params -> same postfix, so samples sharing an
    orientation/slicing reuse the same generated atlas files."""
    orient_part = '_'.join(str(o) for o in orientation)
    if slicing is None:
        slicing_part = 'full'
    else:
        parts = []
        for s in slicing:
            if s.start is None and s.stop is None:
                parts.append('full')
            else:
                parts.append(f'{s.start or 0}-{s.stop}')
        slicing_part = '_'.join(parts)
    return f'{orient_part}__{slicing_part}'

def prepare_annotation():
    """使用从 config 加载的参数"""
    annotation_file, vol_annotation_file, reference_file = ano.prepare_annotation_files(
        slicing=MY_SLICING,
        orientation=MY_ORIENTATION,
        postfix=build_annotation_postfix(MY_ORIENTATION, MY_SLICING),
        overwrite=True,
        verbose=True)
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

# ==== ==== Crop resampled for registration ==== ====
RESAMPLED_FOR_REG = os.path.join(DATA_DIR, 'resampled_cropped.tif')

def crop_resampled_for_registration():
    """Crop resampled image to brain region; returns path to use for registration.
    Coordinates are in voxels of the resampled image (@ 20um).
    Sets CROP_OFFSET so transformation() can correct cell coordinates.
    """
    global CROP_OFFSET
    if CROP_CONFIG is None or all(CROP_CONFIG.get(k) is None for k in ('x', 'y', 'z')):
        print("No crop configured — using full resampled image for registration.")
        return ws.filename('resampled')

    img = io.read(ws.filename('resampled'))  # ClearMap: (X, Y, Z)
    x1, x2 = CROP_CONFIG['x'] if CROP_CONFIG.get('x') else (0, img.shape[0])
    y1, y2 = CROP_CONFIG['y'] if CROP_CONFIG.get('y') else (0, img.shape[1])
    z1, z2 = CROP_CONFIG['z'] if CROP_CONFIG.get('z') else (0, img.shape[2])

    CROP_OFFSET = np.array([x1, y1, z1], dtype=float)
    cropped = img[x1:x2, y1:y2, z1:z2]
    io.write(RESAMPLED_FOR_REG, cropped)
    print(f"Cropped resampled image: {img.shape} -> {cropped.shape}, offset={CROP_OFFSET}")
    return RESAMPLED_FOR_REG

# ==== ==== Alignment ==== ====
#check result.mhd in ws.filename('auto_to_reference') folder
def align_to_reference(): # Align resampled image to reference, save transform params
    fixed_image = crop_resampled_for_registration()
    align_reference_parameter = {
        "moving_image": reference_file, #moving the reference to the sample
        "fixed_image": fixed_image,
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
        sink_shape=io.shape(ws.filename('resampled'))  # always use full resampled shape
    )
    coordinates = coordinates - CROP_OFFSET  # shift into cropped image coordinate system
    coordinates = elx.transform_points(
        coordinates, sink=None,
        transform_directory=ws.filename('auto_to_reference'),
        binary=False,
        indices=True  # voxel coordinates
    )
    return coordinates

def insertdir(parent_file, i, name='cell_registration'):
    """Insert a label directory in file path."""
    dir_inserted = os.path.join(os.path.split(parent_file)[0], name, f'{i}')
    if not os.path.exists(dir_inserted):
        os.makedirs(dir_inserted)
    return os.path.join(dir_inserted, os.path.basename(parent_file))

def process_cell_class(class_name):
    # 1. Load points for this class
    cell_points_file = os.path.join(CELL_CENTROIDS_DIR, f'{CELL_PREFIX}{class_name}.csv')
    if not os.path.exists(cell_points_file):
        print(f"Skipping class {class_name}: File not found.")
        return

    try:
        points = np.loadtxt(cell_points_file, delimiter=',', skiprows=1, usecols=(0, 1, 2))
        
        if points.ndim == 1:
            points = points.reshape(1, -1)
            
    except StopIteration:
        # 应对完全为空的 CSV 文件
        points = np.array([])
    
    if len(points) == 0:
        print(f"Skipping class {class_name}: 0 cells found in csv.")
        return

    # 2. Transform coordinates
    coordinates = points / ratio  
    coordinates_transformed = transformation(coordinates) 
    
    # 3. Annotation (使用内存中的 atlas_volume 数组)
    indices = np.round(coordinates_transformed).astype(int)
    
    sh = atlas_volume.shape 
    
    valid_x = (indices[:, 0] >= 0) & (indices[:, 0] < sh[0])
    valid_y = (indices[:, 1] >= 0) & (indices[:, 1] < sh[1])
    valid_z = (indices[:, 2] >= 0) & (indices[:, 2] < sh[2])
    valid_mask = valid_x & valid_y & valid_z
    
    # 3.3 直接从数组读取 ID
    raw_ids = np.zeros(len(indices), dtype=int) # 默认为 0
    
    if np.any(valid_mask):
        raw_ids[valid_mask] = atlas_volume[
            indices[valid_mask, 0],
            indices[valid_mask, 1],
            indices[valid_mask, 2]
        ]
    
    label_values = np.zeros(len(raw_ids), dtype=int)
    name_values = np.array(['background'] * len(raw_ids), dtype='object')
    
    mask_0 = (raw_ids == 0)
    mask_neg1 = (raw_ids == -1)
    mask_valid_id = ~(mask_0 | mask_neg1)
    
    if np.any(mask_neg1):
        name_values[mask_neg1] = 'no label'
        
    if np.any(mask_valid_id):
        try:
            valid_ids = raw_ids[mask_valid_id]
            label_values[mask_valid_id] = ano.convert_label(valid_ids, key='id', value='graph_order')
            name_values[mask_valid_id] = ano.convert_label(valid_ids, key='id', value='name')
        except Exception as e:
            print(f"Warning: Mapping error in class {class_name}. Details: {e}")
    
    # (结束自定义逻辑) =======================================================

    # 4. Voxelization (density counts tif — full atlas-resolution volume, can be very large)
    if SAVE_DENSITY_TIF:
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
            sink=insertdir(ws.filename('density', postfix='counts'), class_name),
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
    
    io.write(insertdir(ws.filename('cell_registration'), class_name), cells_data)
    np.savetxt(
        insertdir(ws.filename('cell_registration', extension='csv'), class_name), 
        cells_data, delimiter=',', fmt='%s'
    )

print(f"Loading annotation volume from: {annotation_file}")
atlas_volume = io.read(annotation_file)
print(f"Atlas loaded. Shape: {atlas_volume.shape}")

print('Starting cell alignment...')

# 动态获取所有匹配的 CSV 文件
csv_files = glob.glob(os.path.join(CELL_CENTROIDS_DIR, f'{CELL_PREFIX}*.csv'))

print(f"Found {len(csv_files)} cell classes to process.")

for file_path in csv_files:
    filename = os.path.basename(file_path)
    # 动态替换前缀
    class_name = filename.replace(CELL_PREFIX, '').replace('.csv', '')
    
    print(f"\n--- Processing class: {class_name} ---")
    process_cell_class(class_name)

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

