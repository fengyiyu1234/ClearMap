# -*- coding: utf-8 -*-
"""

Script - volume registration and cell mapping using ClearMap2

"""
#  nohup python cellMap.py &> ./logs/02-03-25-16-16.txt &
#nohup python cellMap_fengyi.py &> /data/hdd12tb-1/fengyi/COMBINe/clearmap/fw2/log/registration.txt &
#%% load ClearMap modules
from ClearMap.Environment import *
import csv
import os
import shutil

# stitched = 'name of stitched dataset' like a multi-page tif, e.g.
stitched = 'registration.tif'

# directory = '/path/to/stitched/' e.g.
#directory = '/rsstu/users/t/tghashg/MADMbrains/Ryan/11-07-2024/mahdi_first_p5/numorph/'
directory = '/data/hdd12tb-1/fengyi/COMBINe/clearmap/fw2'


RES = np.array([0.65,0.65,20]) #voxel size of original dataset where the detection results are generated
source_resolution = np.array([5.2, 5.2, 20]) #voxel size of stitched dataset used for registration  
ratio = source_resolution / res 
sink_resolution   = np.array([25, 25, 25]) # voxel size of resampled dataset and atlas used for registration
num_class = 6 # number of classes

ws = wsp.Workspace('CellMap', directory=directory);
ws.update(stitched=stitched)
ws.debug = False

resources_directory = settings.resources_path

ws.info()

#%% init atals and reference files

# adjust x, y, z range and orientation
annotation_file, vol_annotation_file, reference_file, distance_file=ano.prepare_annotation_files(
    slicing=(slice(None),slice(None),slice(0,228)), orientation=(-3,-1,2),  #2: ventral->dorsal, 1: left->right, 3: posterior->anterior
    overwrite=False, verbose=True);

#alignment parameter files    
align_channels_affine_file   = io.join(resources_directory, 'Alignment/align_affine.txt')
align_reference_affine_file  = io.join(resources_directory, 'Alignment/align_affine.txt')
align_reference_bspline_file = io.join(resources_directory, 'Alignment/align_bspline.txt')

#%% resample
# adjust source_resolution of stitched image
resample_parameter = {
    "source_resolution" : tuple(source_resolution), #registration.tif resolution
    "sink_resolution"   : tuple(sink_resolution),
    "processes" : None,
    "verbose" : True,
    };
res.resample(ws.filename('stitched'), sink=ws.filename('resampled'), **resample_parameter)

#%% align stitched to reference
align_reference_parameter = {
    #moving and reference images
    "moving_image" : reference_file,
    "fixed_image"  : ws.filename('resampled'),
    #elastix parameter files for alignment
    "affine_parameter_file"  :  align_reference_affine_file,
    "bspline_parameter_file" :  align_reference_bspline_file,
    #directory of the alignment result
    "result_directory" :  ws.filename('auto_to_reference')
    };

elx.align(**align_reference_parameter);

#%% load detected points (um)

print('Starting cell alignment...')

def transformation(coordinates):

  coordinates = res.resample_points(
                  coordinates, sink=None, orientation=None,
                  source_shape=io.shape(ws.filename('stitched')),
                  sink_shape=io.shape(ws.filename('resampled')));

  coordinates = elx.transform_points(
                  coordinates, sink=None,
                  transform_directory=ws.filename('auto_to_reference'),
                  binary=True, indices=False);

  return coordinates;

def insertdir(dir,i,name='cell_registration'):
  
    dir_inserted = os.path.join(os.path.split(dir)[0],name,'{}'.format(i))
    if not os.path.exists(dir_inserted):
        os.makedirs(dir_inserted)
        
    return os.path.join(dir_inserted, os.path.basename(dir))

for i in range(0,num_class):
    
    with open(os.path.join(directory, 'cell_centroids',
                           'obj_{}.csv'.format(i)), newline='') as csvfile:
        points = np.array(list(csv.reader(csvfile)), dtype=float)

    coordinates = points/ratio/RES; # coordinates in source_resolution
    coordinates_transformed = transformation(coordinates);
    
    #% Cell annotation
    
    label = ano.label_points(coordinates_transformed, annotation_file, key='graph_order');
    names = ano.convert_label(label, key='graph_order', value='name');
    
    #% Unweighted voxelization - cell density
    
    voxelization_parameter = dict(
          shape = io.shape(annotation_file),
          dtype = None,
          weights = None,
          method = 'sphere',
          radius = (1,1,1),
          kernel = None,
          processes = None,
          verbose = True
        )
    
    vox.voxelize(coordinates_transformed,
                 sink=insertdir(ws.filename('density', postfix='counts'),i),
                 **voxelization_parameter);
    
    
    points.dtype=[(c,float) for c in ('x','y','z')]
    coordinates_transformed.dtype=[(t,float) for t in ('xt','yt','zt')]
    label = np.array(label, dtype=[('graph_order', int)]);
    names = np.array(names, dtype=[('name', 'a256')])
    
    import numpy.lib.recfunctions as rfn
    cells_data = rfn.merge_arrays([points, coordinates_transformed, label, names],
                                  flatten=True, usemask=False)
    io.write(insertdir(ws.filename('cell_registration'),i), cells_data)
    
    #% CSV export
    
    np.savetxt(insertdir(ws.filename('cell_registration', extension='csv'),i), cells_data,  delimiter=',', fmt='%s')

#%% Transform annotation volume


path = settings.elastix_path
transformix_binary = os.path.join(path, 'bin/transformix')
# annotation_vol = '/path/to/annotated.mhd' # annotated volume from NuMorph

vol_dir = os.path.join(directory, 'volume')
if not os.path.exists(vol_dir):
    os.makedirs(vol_dir)
shutil.copy2(os.path.join(ws.filename('auto_to_reference'),'TransformParameters.0.txt'), vol_dir)
shutil.copy2(os.path.join(ws.filename('auto_to_reference'),'TransformParameters.1.txt'), vol_dir)
transform_parameter_file = os.path.join(vol_dir,'TransformParameters.1.txt')

# remember to change FinalBSplineInterpolationOrder to 0 when transforming label data
with open(transform_parameter_file, 'r') as file :
  filedata = file.read()
filedata = filedata.replace('FinalBSplineInterpolationOrder 3', 'FinalBSplineInterpolationOrder 0')
with open(transform_parameter_file, 'w') as file:
  file.write(filedata)

# transform annotated volume to dataset
cmd = '%s -in %s -out %s -tp %s' % (transformix_binary, vol_annotation_file, vol_dir, transform_parameter_file);
res = os.system(cmd);
