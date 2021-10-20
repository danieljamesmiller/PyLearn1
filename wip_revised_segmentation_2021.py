# Libraries
import numpy as np
import pandas as pd
import cv2, os, pathlib, sys, math, os.path, json, glob
import imageio, png, statistics, time
from ipywidgets import interact, fixed
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as mpplt
import mayavi
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import nibabel as nib
import nipype
import nipype.algorithms.metrics
from nipype.algorithms.metrics import Distance, Overlap, FuzzyOverlap, ErrorMap
from scipy import misc, stats, ndimage
from scipy.ndimage import label, measurements
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull, convex_hull_plot_2d
import SimpleITK as sitk
import sklearn
from sklearn import preprocessing
from sklearn.cluster import KMeans
import skimage
from skimage import color, data, exposure, filters, img_as_float, img_as_uint, io, measure
from skimage.color import label2rgb, rgb2gray
from skimage.feature import peak_local_max, canny, greycomatrix
from skimage.filters import rank, sobel, scharr, threshold_local, threshold_otsu
from skimage.filters.rank import bottomhat, mean_bilateral, minimum, percentile
from skimage.measure import find_contours
from skimage.morphology import disk, local_minima
from skimage.segmentation import *
from skimage.restoration import *
from skimage.util import img_as_ubyte, invert
from sklearn.metrics import jaccard_score
from pathlib import Path
from PIL import Image
from datetime import datetime
from matplotlib import cm
from collections import OrderedDict
import re, csv
from progress.bar import IncrementalBar
from collections import OrderedDict
import shutil


##############################################
##############################################

# """ FIRST FUNCTION IS A GENERIC OBJECT DETECTOR FOR USE ON IMAGE PATCHES AFTER CHUNKING... """
# def object_detection(image):
#     curr_dir = os.getcwd()
#     # For segmenting stained (black) neurons in light brown tissue -- MGN project
#     #im_rgb = imageio.imread('W312_SeriesBDAd_Section0020_Brain_Whole_000210_xa1000xb1400ya0400yb)
#     im_rgb = imageio.imread(image)
#     metadata = Path(image).stem
#     mdata = '_'.join(metadata.split('_')[:-1])
#     # Convert to HSV space
#     im_hsv = color.convert_colorspace(arr=im_rgb, fromspace='rgb', tospace='hsv')
#     val = im_hsv[:,:,2]
#     # Normalize image lighting
#     claheV = exposure.equalize_adapthist(image=val, kernel_size=10, clip_limit=0.01, nbins=100)
#     # Aggressively smooth image to eliminate background
#     claheV = img_as_ubyte(claheV)
#     bil1V = rank.mean_bilateral(image=claheV, selem=disk(radius=50), s0=100, s1=100)
#     # Set up threshold parameters to eliminate outlier minimal values 
#     mx = np.amax(bil1V)
#     mn = np.amin(bil1V)
#     rng = (mx-mn)
#     thresh = mx - (rng*0.5)
#     # print('max', mx, 'min', mn, 'range', rng, 'threshold', thresh)
#     im = bil1V
#     im[im < thresh] = 0
#     # Aggressively smooth image to eliminate background
#     bilm = rank.mean_bilateral(image=im, selem=disk(radius=50), s0=100, s1=100)
#     # Merge nearby local minimum values, and test array range for object (!!!)
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
#     seeds = img_as_ubyte(cv2.dilate(bilm, kernel, iterations=5))
#     mx1 = np.max(seeds)
#     mn1 = np.min(seeds)
#     rng1 = mx1-mn1
#     if rng1 > 100:
#         # Set nomenclature
#         out_fname = mdata + '_Object.png' 
#         out_dir = '/Users/djm/Desktop/2018_python/test_targets/'
#         # If object is present, write contours to file
#         kernel_m = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
#         mask = img_as_ubyte(cv2.dilate(bilm, kernel_m, iterations=1))
#         contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#         prediction = cv2.drawContours(mask, contours, -1, (255, 0, 0), cv2.FILLED)
#         pred = invert(prediction)
#         im0 = Image.fromarray(pred)
#         im0.save(out_fname)
#         # Now, move both files to folder containing target segmentations
#         shutil.move(src=out_fname, dst=out_dir)
#         shutil.move(src=image, dst=out_dir)
#     else:
#         # # WIP ...

#         # Set nomenclature
#         out_fname = mdata + '_Blank.png'
#         out_dir = '/Users/djm/Desktop/2018_python/test_blanks/'
        
        
        
#         ## REVISED THIS TO BE A LARGER PATCH
#         mask = np.zeros([400,400], dtype=np.uint32)
#         im0 = Image.fromarray(mask)
#         im0.save(out_fname)
#         # Now, move both files to folder containing blank segmentations
#         shutil.move(src=out_fname, dst=out_dir)
#         shutil.move(src=image, dst=out_dir)


####################################
####################################
####################################
###### Start Object Chunking #######
####################################
####################################
####################################


""" Second, we can ingest the inventory, and begin chunking its contents """
# Read in the prepared datafile containing an inventory of the mosaic files to be processed
# Establish nomenclature
cdir = os.getcwd()
run_location = os.getcwd().split('/')[-1:][0]
search0 = run_location + '_InventoryMosaic.csv'
# Imported df [has 'container' , 'fname' , 'fpath']
df1 = pd.read_csv(search0)
# make a list of all the unique containers
container_list = sorted(df1['container'].unique().tolist())
completed = []
"""Loop through the containers to get global coordinate nomenclature"""
for item in container_list:
    if item not in completed:
        completed.append(item)
    else:
        continue
    # First, get the Assembly Data file and process it...
    search1 = item + '_AssemblyData.txt'
    df_temp = df1.loc[df1['fname'] == search1].copy()
    # Ignore empty frames
    if len(df_temp) < 1:
        continue
    # Get first file path for container mosaic data
    assemble_fpath = df_temp['fpath'].values[0]
    # Run first function
    assembly_txt_conversion(assemble_fpath)
    # Next file path
    reorder_fpath = item + '_AssemblyConverted.csv'
    # Run next function
    reorder_assembly_data(reorder_fpath)
    # Final file path
    ingest_fpath = item + '_AssemblyModified.csv'
    # Run final function
    ingest_assembly_data(ingest_fpath)
    # Second, read-in the transformed data for each container, and loop through images 
    search2 = item + '_AssemblyGlobal.csv'
    df2 = pd.read_csv(search2)
    ln = len(df2)
    bar = IncrementalBar('Processing:', max=ln)
    # Loop through the images associated with each container's datafile
    for index, row in df2.iterrows():
        # For now, when testing, simply write out the file order (do not process files)
        # df2 contains: 'original_tile_id', 'original_tile_path', 'new_global_id' of each original tile image
        fname = row.original_tile_path
        original_tile_id = row.original_tile_id
        global_tile_id = row.new_global_id
        # Set up nomenclature for subarray output
        mdata = row.new_global_id.split('_')
        # Take all but x and y
        metadata = '_'.join(mdata[:-2])
        # Assign x and y coordinate data to other
        x_coord = int(mdata[-1][1:])
        y_coord = int(mdata[-2][1:])
        # Read-in the file, after setting nomenclature
        img0 = imageio.imread(fname)
        # Set the dimensions of the chunking procedure,
        # Specifically, derive cut-off values for beginning/edu
        xdim = img0.shape[1]
        ydim = img0.shape[0]
        # Divide image dimension by patch dimension, and keep the number of
        # whole patches in each image dimension (i.e. drop the remainder)
        xnum = xdim/400
        ynum = ydim/400
        # Recall that indexing from 0 and using this as a length increases its magnitude by 1...
        xlim = int(xnum)
        ylim = int(ynum)
        xstop = xlim + 1
        ystop = ylim + 1
        # Some nomenclature
        csv_outpath = metadata + '_GlobalCoordinates.csv'
        deep_tile = metadata + '.npy'
        deep_path = os.path.join(cdir, deep_tile)
        deep_archive = metadata + '.npz'
        # Begin loop of image chunking grid by Y and X values
        # Loop through range of Y values
        # docs for fnx -- range(start, stop, interval)
        for x in range(0, xstop, 1):
            # Define chunking dimensions
            # where (x0 = left, x1 = right; y0 = upper, y1 = lower)
            # Recall: Global coordinate embedded as nomenclature must reflect x|y_coord(s)
            # First option
            if x == 0:
                left = 0
                Global_Left = left + x_coord
                right = 400
                for y in range(0, ystop, 1):
                    if y == 0:
                        upper = 0
                        Global_Upper = upper + y_coord
                        lower = 400
                        # Set coordinate values to being larger bc of global coordinate space
                        GlobalX = 'x' + '{:0>6d}'.format(Global_Left)
                        GlobalY = 'y' + '{:0>6d}'.format(Global_Upper)
                        # Set up output nomenclature
                        out_fname = metadata + '_' + GlobalY + '_' + GlobalX
                        # Set up dictionary
                        nd = {}
                        nd['container'] = metadata
                        nd['id_OriginalTile'] = original_tile_id
                        nd['id_GlobalTile'] = global_tile_id
                        nd['Patch_posX'] = x 
                        nd['Patch_posY'] = y 
                        nd['id_GlobalPatch'] = out_fname
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        if os.path.exists(csv_outpath):
                            new_df.to_csv(csv_outpath, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(csv_outpath, header=True, index=False)
                        # Set up patch creation  
                        patch = img0[upper:lower, left:right]
                        # This is the first patch in the stack, so simply save the file
                        np.save(deep_path, patch)
                        # # # # Potentially now call the object detection pipeline on the subarray patches...
                        # #
                        
                    elif (y > 0 and y < ylim):
                        upper = y*400
                        Global_Upper = upper + y_coord
                        lower = upper + 400
                        # Set coordinate values to being larger bc of global coordinate space
                        GlobalX = 'x' + '{:0>6d}'.format(Global_Left)
                        GlobalY = 'y' + '{:0>6d}'.format(Global_Upper)
                        # Set up output nomenclature
                        out_fname = metadata + '_' + GlobalY + '_' + GlobalX
                        # Set up dictionary
                        nd = {}
                        nd['container'] = metadata
                        nd['id_OriginalTile'] = original_tile_id
                        nd['id_GlobalTile'] = global_tile_id
                        nd['Patch_posX'] = x 
                        nd['Patch_posY'] = y 
                        nd['id_GlobalPatch'] = out_fname
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        if os.path.exists(csv_outpath):
                            new_df.to_csv(csv_outpath, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(csv_outpath, header=True, index=False)
                        # Set up patch creation 
                        patch = img0[upper:lower, left:right]
                        # Check if deep array is present to append, otherwise, create new
                        if os.path.exists(deep_path):
                            deep_arr = np.load(deep_tile)
                            deeper_arr = np.dstack([deep_arr, patch])
                            # Write deeper_arr to file
                            np.save(deep_path, deeper_arr)
                        else:
                            # Otherwise, write current array to file...
                            np.save(deep_path, patch)
                    elif y == ylim:
                        upper = 1792
                        Global_Upper = upper + y_coord
                        lower = 2192
                        # Set coordinate values to being larger bc of global coordinate space
                        GlobalX = 'x' + '{:0>6d}'.format(Global_Left)
                        GlobalY = 'y' + '{:0>6d}'.format(Global_Upper)
                        # Set up output nomenclature
                        out_fname = metadata + '_' + GlobalY + '_' + GlobalX
                        # Set up dictionary
                        nd = {}
                        nd['container'] = metadata
                        nd['id_OriginalTile'] = original_tile_id
                        nd['id_GlobalTile'] = global_tile_id
                        nd['Patch_posX'] = x 
                        nd['Patch_posY'] = y 
                        nd['id_GlobalPatch'] = out_fname
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        if os.path.exists(csv_outpath):
                            new_df.to_csv(csv_outpath, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(csv_outpath, header=True, index=False)
                        # Set up patch creation
                        patch = img0[upper:lower, left:right]
                        # Check if deep array is present to append, otherwise, create new
                        if os.path.exists(deep_path):
                            deep_arr = np.load(deep_tile)
                            deeper_arr = np.dstack([deep_arr, patch])
                            # Write deeper_arr to file
                            np.save(deep_path, deeper_arr)
                        else:
                            # Otherwise, write current array to file...
                            np.save(deep_path, patch)
                    else:
                        continue
            # Second option
            elif (x > 0 and x < xlim):
                left = x*400
                #Global_Left = left + x_coord
                Global_Left = left
                right = left + 400
                for y in range(0, ystop, 1):
                    if y == 0:
                        upper = 0
                        Global_Upper = upper + y_coord
                        lower = 400
                        # Set coordinate values to being larger bc of global coordinate space
                        GlobalX = 'x' + '{:0>6d}'.format(Global_Left)
                        GlobalY = 'y' + '{:0>6d}'.format(Global_Upper)
                        # Set up output nomenclature
                        out_fname = metadata + '_' + GlobalY + '_' + GlobalX
                        # Set up dictionary
                        nd = {}
                        nd['container'] = metadata
                        nd['id_OriginalTile'] = original_tile_id
                        nd['id_GlobalTile'] = global_tile_id
                        nd['Patch_posX'] = x 
                        nd['Patch_posY'] = y 
                        nd['id_GlobalPatch'] = out_fname
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        if os.path.exists(csv_outpath):
                            new_df.to_csv(csv_outpath, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(csv_outpath, header=True, index=False)
                        # Set up patch creation  
                        patch = img0[upper:lower, left:right]
                        # Check if deep array is present to append, otherwise, create new
                        if os.path.exists(deep_path):
                            deep_arr = np.load(deep_tile)
                            deeper_arr = np.dstack([deep_arr, patch])
                            # Write deeper_arr to file
                            np.save(deep_path, deeper_arr)
                        else:
                            # Otherwise, write current array to file...
                            np.save(deep_path, patch)
                    elif (y > 0 and y < ylim):
                        upper = y*400
                        Global_Upper = upper + y_coord
                        lower = upper + 400
                        # Set coordinate values to being larger bc of global coordinate space
                        GlobalX = 'x' + '{:0>6d}'.format(Global_Left)
                        GlobalY = 'y' + '{:0>6d}'.format(Global_Upper)
                        # Set up output nomenclature
                        out_fname = metadata + '_' + GlobalY + '_' + GlobalX
                        # Set up dictionary
                        nd = {}
                        nd['container'] = metadata
                        nd['id_OriginalTile'] = original_tile_id
                        nd['id_GlobalTile'] = global_tile_id
                        nd['Patch_posX'] = x 
                        nd['Patch_posY'] = y 
                        nd['id_GlobalPatch'] = out_fname
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        if os.path.exists(csv_outpath):
                            new_df.to_csv(csv_outpath, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(csv_outpath, header=True, index=False)
                        # Set up patch creation 
                        patch = img0[upper:lower, left:right]
                        # Check if deep array is present to append, otherwise, create new
                        if os.path.exists(deep_path):
                            deep_arr = np.load(deep_tile)
                            deeper_arr = np.dstack([deep_arr, patch])
                            # Write deeper_arr to file
                            np.save(deep_path, deeper_arr)
                        else:
                            # Otherwise, write current array to file...
                            np.save(deep_path, patch)
                    elif y == ylim:
                        upper = 1792
                        Global_Upper = upper + y_coord
                        lower = 2192
                        # Set coordinate values to being larger bc of global coordinate space
                        GlobalX = 'x' + '{:0>6d}'.format(Global_Left)
                        GlobalY = 'y' + '{:0>6d}'.format(Global_Upper)
                        # Set up output nomenclature
                        out_fname = metadata + '_' + GlobalY + '_' + GlobalX
                        # Set up dictionary
                        nd = {}
                        nd['container'] = metadata
                        nd['id_OriginalTile'] = original_tile_id
                        nd['id_GlobalTile'] = global_tile_id
                        nd['Patch_posX'] = x 
                        nd['Patch_posY'] = y 
                        nd['id_GlobalPatch'] = out_fname
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        if os.path.exists(csv_outpath):
                            new_df.to_csv(csv_outpath, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(csv_outpath, header=True, index=False)
                        # Set up patch creation
                        patch = img0[upper:lower, left:right]
                        # Check if deep array is present to append, otherwise, create new
                        if os.path.exists(deep_path):
                            deep_arr = np.load(deep_tile)
                            deeper_arr = np.dstack([deep_arr, patch])
                            # Write deeper_arr to file
                            np.save(deep_path, deeper_arr)
                        else:
                            # Otherwise, write current array to file...
                            np.save(deep_path, patch)
                    else:
                        continue
            # Third option
            elif x == xlim:
                left = 2352
                #Global_Left = left + x_coord
                Global_Left = left
                right = 2752
                for y in range(0, ystop, 1):
                    if y == 0:
                        upper = 0
                        Global_Upper = upper + y_coord
                        lower = 400
                        # Set coordinate values to being larger bc of global coordinate space
                        GlobalX = 'x' + '{:0>6d}'.format(Global_Left)
                        GlobalY = 'y' + '{:0>6d}'.format(Global_Upper)
                        # Set up output nomenclature
                        out_fname = metadata + '_' + GlobalY + '_' + GlobalX
                        # Set up dictionary
                        nd = {}
                        nd['container'] = metadata
                        nd['id_OriginalTile'] = original_tile_id
                        nd['id_GlobalTile'] = global_tile_id
                        nd['Patch_posX'] = x 
                        nd['Patch_posY'] = y 
                        nd['id_GlobalPatch'] = out_fname
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        if os.path.exists(csv_outpath):
                            new_df.to_csv(csv_outpath, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(csv_outpath, header=True, index=False)
                        # Set up patch creation  
                        patch = img0[upper:lower, left:right]
                        # Check if deep array is present to append, otherwise, create new
                        if os.path.exists(deep_path):
                            deep_arr = np.load(deep_tile)
                            deeper_arr = np.dstack([deep_arr, patch])
                            # Write deeper_arr to file
                            np.save(deep_path, deeper_arr)
                        else:
                            # Otherwise, write current array to file...
                            np.save(deep_path, patch)
                    elif (y > 0 and y < ylim):
                        upper = y*400
                        Global_Upper = upper + y_coord
                        lower = upper + 400
                        # Set coordinate values to being larger bc of global coordinate space
                        GlobalX = 'x' + '{:0>6d}'.format(Global_Left)
                        GlobalY = 'y' + '{:0>6d}'.format(Global_Upper)
                        # Set up output nomenclature
                        out_fname = metadata + '_' + GlobalY + '_' + GlobalX
                        # Set up dictionary
                        nd = {}
                        nd['container'] = metadata
                        nd['id_OriginalTile'] = original_tile_id
                        nd['id_GlobalTile'] = global_tile_id
                        nd['Patch_posX'] = x 
                        nd['Patch_posY'] = y 
                        nd['id_GlobalPatch'] = out_fname
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        if os.path.exists(csv_outpath):
                            new_df.to_csv(csv_outpath, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(csv_outpath, header=True, index=False)
                        # Set up patch creation 
                        patch = img0[upper:lower, left:right]
                        # Check if deep array is present to append, otherwise, create new
                        if os.path.exists(deep_path):
                            deep_arr = np.load(deep_tile)
                            deeper_arr = np.dstack([deep_arr, patch])
                            # Write deeper_arr to file
                            np.save(deep_path, deeper_arr)
                        else:
                            # Otherwise, write current array to file...
                            np.save(deep_path, patch)
                    elif y == ylim:
                        upper = 1792
                        Global_Upper = upper + y_coord
                        lower = 2192
                        # Set coordinate values to being larger bc of global coordinate space
                        GlobalX = 'x' + '{:0>6d}'.format(Global_Left)
                        GlobalY = 'y' + '{:0>6d}'.format(Global_Upper)
                        # Set up output nomenclature
                        out_fname = metadata + '_' + GlobalY + '_' + GlobalX
                        # Set up dictionary
                        nd = {}
                        nd['container'] = metadata
                        nd['id_OriginalTile'] = original_tile_id
                        nd['id_GlobalTile'] = global_tile_id
                        nd['Patch_posX'] = x 
                        nd['Patch_posY'] = y 
                        nd['id_GlobalPatch'] = out_fname
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        if os.path.exists(csv_outpath):
                            new_df.to_csv(csv_outpath, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(csv_outpath, header=True, index=False)
                        # Set up patch creation
                        patch = img0[upper:lower, left:right]
                        # Check if deep array is present to append, otherwise, create new
                        if os.path.exists(deep_path):
                            deep_arr = np.load(deep_tile)
                            deeper_arr = np.dstack([deep_arr, patch])
                            # Write deeper_arr to file
                            np.save(deep_path, deeper_arr)
                        else:
                            # Otherwise, write current array to file...
                            np.save(deep_path, patch)
                    else:
                        continue
            else:
                continue