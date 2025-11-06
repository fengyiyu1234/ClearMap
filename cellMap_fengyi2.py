# -*- coding: utf-8 -*-
"""
Script for volume registration and cell mapping using ClearMap2.
Organized for clarity and maintainability.
"""
#nohup python cellMap_fengyi2.py &> /data/hdd12tb-1/fengyi/COMBINe/clearmap/fw2/log/registration.txt &
import os
import csv
import shutil
import numpy as np

# ==== === Project/Experiment Parameters === ===
# ----- File and Path Setting -----
STITCHED_FILENAME = 'registration.tif'
DATA_DIR = '/data/hdd12tb-1/fengyi/COMBINe/clearmap/fw2'
CELL_CENTROIDS_DIR = os.path.join(DATA_DIR, 'cell_centroids')
RESOURCES_DIR = None # will be set later by settings.resources_path

# ===== Data-specific Parameters -----
VOXEL_SIZE_ORIGINAL = np.array([0.65, 0.65, 20])      # detection resolution
VOXEL_SIZE_STITCHED = np.array([5.2, 5.2, 20])        # stitched dataset (pre-reg)
VOXEL_SIZE_RESAMPLED = np.array([25, 25, 25])         # resampled/atlas
N_CLASSES = 6                                         # number of cell classes

# ==== ==== ClearMap Modules ==== ====
from ClearMap.Environment import *
import numpy.lib.recfunctions as rfn

# --- Derived params
ratio = VOXEL_SIZE_STITCHED / VOXEL_SIZE_ORIGINAL

# ==== ==== Workspace Setup ==== ====
ws = wsp.Workspace('CellMap', directory=DATA_DIR)
ws.update(stitched=STITCHED_FILENAME)
ws.debug = False
RESOURCES_DIR = settings.resources_path
ws.info()

# ==== ==== Annotation & Reference Preparation ==== ====
def prepare_annotation():
    """Adjust and prepare annotation/reference files."""
    annotation_file, vol_annotation_file, reference_file, distance_file = ano.prepare_annotation_files(
        slicing=(slice(None),slice(None),slice(None)), orientation=(-3,-1,-2),
        overwrite=False, verbose=True)
    return annotation_file, vol_annotation_file, reference_file, distance_file

(annotation_file, vol_annotation_file, 
 reference_file, distance_file) = prepare_annotation()

# Alignment parameter files
ALIGNMENT_PATH = os.path.join(RESOURCES_DIR, 'Alignment')
align_channels_affine_file   = io.join(ALIGNMENT_PATH, 'align_affine.txt')
align_reference_affine_file  = io.join(ALIGNMENT_PATH, 'align_affine.txt')
align_reference_bspline_file = io.join(ALIGNMENT_PATH, 'align_bspline.txt')

# ==== ==== Resampling ==== ====
def resample_stitched():
    resample_parameter = {
        "source_resolution": tuple(VOXEL_SIZE_STITCHED),
        "sink_resolution": tuple(VOXEL_SIZE_RESAMPLED),
        "processes": None,
        "verbose": True,
    }
    res.resample(ws.filename('stitched'), sink=ws.filename('resampled'), **resample_parameter)

resample_stitched()

# ==== ==== Alignment ==== ====
def align_to_reference(): # Align resampled image to reference, save transform params
    align_reference_parameter = {
        "moving_image": reference_file, 
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
    coordinates = res.resample_points( #no change in orientation
        coordinates, sink=None, orientation=None, 
        source_shape=io.shape(ws.filename('stitched')),
        sink_shape=io.shape(ws.filename('resampled')) #downsample
    )
    coordinates = elx.transform_points(
        coordinates, sink=None,
        transform_directory=ws.filename('auto_to_reference'), #transform to reference space 
        binary=True, indices=False
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
    cell_points_file = os.path.join(CELL_CENTROIDS_DIR, f'obj_{class_idx}.csv')
    with open(cell_points_file, newline='') as csvfile:
        points = np.array(list(csv.reader(csvfile)), dtype=float)
    # 2. Transform coordinates
    coordinates = points / ratio/ VOXEL_SIZE_ORIGINAL  # goes to stitched resolution
    coordinates_transformed = transformation(coordinates)
    # 3. Annotation
    label = ano.label_points(coordinates_transformed, annotation_file, key='graph_order')
    names = ano.convert_label(label, key='graph_order', value='name')
    # 4. Voxelization
    voxelization_parameter = dict(
        shape=io.shape(annotation_file),
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
    # 5. Save results (npz and csv)
    points.dtype = [(c, float) for c in ('x', 'y', 'z')]
    coordinates_transformed.dtype = [(t, float) for t in ('xt', 'yt', 'zt')]
    label = np.array(label, dtype=[('graph_order', int)])
    names = np.array(names, dtype=[('name', 'a256')])
    cells_data = rfn.merge_arrays([points, coordinates_transformed, label, names], flatten=True, usemask=False)
    io.write(insertdir(ws.filename('cell_registration'), class_idx), cells_data)
    np.savetxt(
        insertdir(ws.filename('cell_registration', extension='csv'), class_idx), 
        cells_data, delimiter=',', fmt='%s'
    )

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

