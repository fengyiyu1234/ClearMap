# -*- coding: utf-8 -*-
"""
Script for volume registration and cell mapping using ClearMap2.
Organized for clarity and maintainability.
"""
#make sure input points are in pixel coordinates of the stitched image
#if spatial coordinate, transform_points(indices=False)
# The script logs to <data_dir>/log.txt on its own (see "Logging" below), so
# an explicit &> redirect is no longer needed:
# nohup python cellMap.py --config config_12t.yaml &
import os
# Headless servers have no X11 display, but ClearMap.Environment pulls in
# Qt-based plotting modules on import (unused here); force the offscreen
# Qt backend so that import doesn't crash. Doesn't override if already set.
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import csv
import shutil
import numpy as np
import pandas as pd
import tifffile
from ClearMap.Environment import *
import numpy.lib.recfunctions as rfn
import glob
import yaml
import sys
import argparse

# ==== 1. 加载配置文件 ====
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_args():
    default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
    parser = argparse.ArgumentParser(description='Run the ClearMap registration + cell mapping pipeline for one sample.')
    parser.add_argument('--config', default=default_config,
                         help=f'Path to the sample config YAML (default: {default_config})')
    # argparse chokes on stray args from e.g. `python -m ipykernel ...`; ignore unknowns.
    args, _ = parser.parse_known_args()
    return args

cfg = load_config(parse_args().config)

# ==== 2. 映射参数 ====
DATA_DIR = cfg['paths']['data_dir']

# ==== ==== Logging ==== ====
# Mirror stdout/stderr into <data_dir>/log.txt, so the log always lands next
# to the sample it belongs to (no need to remember/type the full data_dir
# path in a shell redirect every time). Appends, so re-runs on the same
# sample accumulate history instead of clobbering the previous log.
class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def fileno(self):
        # subprocess.Popen(stdout=sys.stdout, ...) (used by Elastix.align()
        # to run the elastix binary) needs a real OS file descriptor, which
        # a pure-Python object can't provide for both streams at once.
        # Delegate to the first stream (the real terminal/nohup stdout) so
        # the elastix subprocess keeps working; its own output is still
        # fully captured in <result_directory>/elastix.log regardless, so
        # nothing is lost — it's just not duplicated into our log.txt too.
        return self.streams[0].fileno()

LOG_PATH = os.path.join(DATA_DIR, 'log.txt')
_log_file = open(LOG_PATH, 'a')
print(f"\n{'='*30}\nRun started: {pd.Timestamp.now()}\n{'='*30}", file=_log_file, flush=True)
sys.stdout = _Tee(sys.stdout, _log_file)
sys.stderr = _Tee(sys.stderr, _log_file)
print(f"Logging to {LOG_PATH}")

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

    # Brain mask for the fixed (sample) image, so the MI metric isn't
    # diluted by background — matters most for small/misshapen samples.
    # Generated on the exact image passed as -f, so shapes always match.
    fixed_mask_path = os.path.join(DATA_DIR, 'resampled_mask.tif')
    bmask.generate_brain_mask(fixed_image, sink_path=fixed_mask_path)

    align_reference_parameter = {
        "moving_image": reference_file, #moving the reference to the sample
        "fixed_image": fixed_image,
        "affine_parameter_file": align_reference_affine_file,
        "bspline_parameter_file": align_reference_bspline_file,
        "result_directory": ws.filename('auto_to_reference'),
        "fixed_mask": fixed_mask_path,
    }
    elx.align(**align_reference_parameter)

align_to_reference()

# ==== ==== Cell Points Transformation & Annotation ==== ====
def transformation(coordinates):
    """Resample & transform coordinates to reference space.

    Returns both the resampled (pre-registration) coordinates and the
    final atlas-registered coordinates.
    """
    resampled = res.resample_points(
        coordinates, sink=None, orientation=None,
        source_shape=io.shape(ws.filename('stitched')),
        sink_shape=io.shape(ws.filename('resampled'))  # always use full resampled shape
    )
    coordinates = resampled - CROP_OFFSET  # shift into cropped image coordinate system
    coordinates = elx.transform_points(
        coordinates, sink=None,
        transform_directory=ws.filename('auto_to_reference'),
        binary=False,
        indices=True  # voxel coordinates
    )
    return resampled, coordinates

def insertdir(parent_file, i, name='cell_registration'):
    """Insert a label directory in file path."""
    dir_inserted = os.path.join(os.path.split(parent_file)[0], name, f'{i}')
    if not os.path.exists(dir_inserted):
        os.makedirs(dir_inserted)
    return os.path.join(dir_inserted, os.path.basename(parent_file))

def _read_centroid_csv(csv_path):
    """Read a cell_centroids/ob_<class>.csv file, tolerating both the
    run_inference.py header format (cx,cy,z,score,slice_name,tile_name --
    see brain_detector/scripts/run_inference.py) and older header-less
    x,y,z-only files. Always returns a DataFrame with cx/cy/z plus
    score/slice_name/tile_name (NaN-filled if absent from the source)."""
    expected = ['cx', 'cy', 'z', 'score', 'slice_name', 'tile_name']
    df = pd.read_csv(csv_path)
    cols = [str(c).strip().lower() for c in df.columns]
    if cols[:3] not in (['cx', 'cy', 'z'], ['x', 'y', 'z']):
        # First row wasn't actually a header -- re-read raw and name
        # columns positionally.
        df = pd.read_csv(csv_path, header=None)
        df.columns = expected[:df.shape[1]]
    else:
        df.columns = cols
        if df.columns[0] == 'x':
            df = df.rename(columns={'x': 'cx', 'y': 'cy'})
    for col in ('score', 'slice_name', 'tile_name'):
        if col not in df.columns:
            df[col] = np.nan
    return df

def process_cell_class(class_name):
    # 1. Load points for this class
    cell_points_file = os.path.join(CELL_CENTROIDS_DIR, f'{CELL_PREFIX}{class_name}.csv')
    if not os.path.exists(cell_points_file):
        print(f"Skipping class {class_name}: File not found.")
        return

    try:
        df_src = _read_centroid_csv(cell_points_file)
    except Exception as e:
        # 应对完全为空 / 损坏的 CSV 文件
        print(f"Skipping class {class_name}: failed to read CSV ({e}).")
        return

    if len(df_src) == 0:
        print(f"Skipping class {class_name}: 0 cells found in csv.")
        return

    # np.ascontiguousarray: DataFrame.values can hand back an F-contiguous
    # array, which silently breaks the .view() struct trick used below to
    # write cells_data -- np.loadtxt (the old reader) never had this problem
    # since it always returns C-contiguous arrays.
    points = np.ascontiguousarray(df_src[['cx', 'cy', 'z']].values, dtype=float)
    slice_names = df_src['slice_name'].fillna('').astype(str).values
    tile_names = df_src['tile_name'].fillna('').astype(str).values
    scores = pd.to_numeric(df_src['score'], errors='coerce').fillna(0.0).values.astype(float)

    # 2. Transform coordinates
    coordinates = points / ratio
    coordinates_resampled, coordinates_transformed = transformation(coordinates)
    
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
    # coordinates_resampled.dtype = [(c, float) for c in ('xr', 'yr', 'zr')]
    # coordinates_transformed.dtype = [(t, float) for t in ('xt', 'yt', 'zt')]
    points = np.ascontiguousarray(points).view([(c, float) for c in ('x', 'y', 'z')])
    coordinates_resampled = np.ascontiguousarray(coordinates_resampled).view([(c, float) for c in ('xr', 'yr', 'zr')])
    coordinates_transformed = np.ascontiguousarray(coordinates_transformed).view([(t, float) for t in ('xt', 'yt', 'zt')])

    label_struct = np.array(label_values, dtype=[('graph_order', int)])
    names_struct = np.array(name_values, dtype=[('name', 'S256')])
    # Tile/slice/score provenance from the source cell_centroids file (see
    # brain_detector/scripts/run_inference.py) -- carried straight through so
    # a cell flagged as suspicious in the viewer can be traced back to its
    # source TB-scale tile without depending on cell_centroids/ still
    # existing alongside this registration output. Column order matters:
    # stats_img_vis_ui.py's load_cells_atlas_df reads these positionally at
    # indices 11 (slice_name), 12 (tile_name), 13 (score).
    slice_struct = np.array(slice_names, dtype=[('slice_name', 'S256')])
    tile_struct = np.array(tile_names, dtype=[('tile_name', 'S256')])
    score_struct = np.array(scores, dtype=[('score', float)])

    cells_data = rfn.merge_arrays(
        [points, coordinates_resampled, coordinates_transformed, label_struct, names_struct,
         slice_struct, tile_struct, score_struct],
        flatten=True, usemask=False
    )

    io.write(insertdir(ws.filename('cell_registration'), class_name), cells_data)

    # CSV written via pandas (not np.savetxt) so slice_name/tile_name save as
    # plain text instead of numpy's b'...' bytes-repr for string fields --
    # this file is meant to be opened directly to trace a cell back to its
    # source tile/slice. Kept header-less (as before) for backward
    # compatibility with stats_img_vis_ui.py's positional CSV reader.
    csv_df = pd.DataFrame(cells_data)
    for col in ('name', 'slice_name', 'tile_name'):
        csv_df[col] = csv_df[col].apply(lambda v: v.decode('utf-8', errors='replace') if isinstance(v, bytes) else v)
    csv_df.to_csv(
        insertdir(ws.filename('cell_registration', extension='csv'), class_name),
        index=False, header=False
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

