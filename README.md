[ClearMap](https://github.com/ChristophKirst/ClearMap2) (Python) is adopted for dataset registration. 

## Installation
Downloaed the codes.  
Replace `./ClearMap/Alignment/Annotation.py`.  
Replace files in `./ClearMap/Resources/Atlas/`.  

Check [ClearMap2 installation guide](https://christophkirst.github.io/ClearMap2Documentation/html/installation.html).  

## Registration

 - Input:
     - Low-resolution (~ 25 µm) stitched volume (.tif) *(from **Stitching**)*.  
     - Stitched cell centroids (.csv) per class *(from **Cell stitching**)*.   
 - Output:  
     - Annotated cell centroids (.csv) per class *(used for **Analysis**)*.  
     - Registered annotation volume (.mhd) *(used for **Analysis**)*.
     - Voxel-wise cell density counts (.tif) per class *(used for **Analysis**)*.  
     
Put the low-resolution stitched volume (.tif) in the dataset root directory, e.g., in `P30MEMXcreMADMEGFR++L/`.  
Follow `cellMap.py` to register the dataset to Allen Brain Atlas and map the detected cells.  
