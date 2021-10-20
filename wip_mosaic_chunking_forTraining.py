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
#from skimage.segmentation import *
#from skimage.restoration import *
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
##### DEFINED FUNCTIONS FOR THE PIPELINE #####
##############################################
##############################################




def make_mosaic_inventory():
    # Set up output filename
    top_dir = os.getcwd()
    run_location = top_dir.split('/')[-1:][0]
    # The extension to search for
    extensions = ['.jpx', '.txt']
    # First, cataloge the directory of images and their 
    # paths (i.e. the file, and each one's location for retrieval)
    for dirpath, dirnames, files in os.walk(top_dir):
        ln = len(files)
        bar = IncrementalBar('Processing:', max=ln)
        for name in files:
            if name.lower().endswith(tuple(extensions)):
                if '._' in name:
                    continue
                if '_CompilerResults' in name:
                    continue
                if '_AssemblyComplete' in name:
                    continue
                item_path = os.path.join(dirpath, name)
                mdata = Path(name).stem
                container = '_'.join(mdata.split('_')[:-1])
                # Nomenclature for output
                nd = {}
                nd['container'] = container
                nd['fpath'] = item_path
                nd['fname'] = name
                # Make new dataframe with output
                df1 = pd.DataFrame().append(nd, ignore_index=True)
                out_fname = run_location + '_InventoryMosaic.csv'
                if os.path.exists(out_fname):
                    df1.to_csv(out_fname, mode='a', header=False, index=False)
                else:
                    df1.to_csv(out_fname, header=True, index=False)
            bar.next()
        bar.finish()

##############################################

def assembly_txt_conversion(input_file):
    """ Convert Assembly Data File Type """
    # Convert assembly.txt to assembly.csv 
    # The goal here is to make accessible all the data from the Assembly file
    core_id = Path(input_file).stem
    core_id = '_'.join(core_id.split('_')[:-1])
    txt_file = input_file
    csv_file = core_id + '_AssemblyConverted.csv'
    with open(txt_file, 'rt') as infile, open(csv_file, 'w+') as outfile:
        stripped = (line.strip() for line in infile)
        lines = (line.split(",") for line in stripped if line)
        writer = csv.writer(outfile)
        writer.writerows(lines)

##############################################

def reorder_assembly_data(input_data):
    """ Ingest the Assembly Data """
    # First, get the container metadata
    mdata = Path(input_data).stem
    mdat = mdata.split('_')[:-1]
    mdt = '_'.join(mdat)
    # Split the original into parts (and ignore the end) 
    # Read in the converted original file
    df0 = pd.read_csv(input_data, header=None)
    # Split into relevant parts
    # Constant size for end part of file data
    # The last four rows are the footer
    df_end = df0[-4:]
    # Constant size for top part of file data
    # The first 27 rows are the header
    df_metadata = df0[0:26]
    # Variable size of coordinate data part of file data
    # Here, defined as portion not including header AND footer (will output df of whatever size)
    df_data = df0[27:-4]
    """  Transform converted file of tile location data """
    # Ideally, would split image delta values from tile data...
    # First, make a list of all the Tiles... after converting to dataframe
    for index, row in df_data.iterrows():
        # Ignore the "delta" file data for now
        if 'Delta' in row[0]:
            continue
        else:
            if ' = ' in row[0]:
                data = row[0].split(' = ')
                label = data[0] 
                value = int(data[1])
                tile_var = label[:-1]
                coord_label = str(label[-1])
                # Adjust index for naming conventions
                tile_no = int(tile_var[4:]) + 1
                tile_num = '{:0>6d}'.format(tile_no)
                coord_val = '{:0>6d}'.format(value)
                tile_id = mdt + '_' + tile_num + '.jpx'
                # Set up naming for output:
                nd = {}
                nd['tile_id'] = tile_id
                nd['coord_label'] = coord_label
                nd['coord_value'] = coord_val
                new_df = pd.DataFrame().append(nd, ignore_index=True)
                output_path = mdt + '_AssemblyModified.csv'
                if os.path.exists(output_path):
                    new_df.to_csv(output_path, mode='a', header=False, index=False)
                else:
                    new_df.to_csv(output_path, header=True, index=False)

##############################################

def ingest_assembly_data(in_data):
    # First, get the container metadata
    mdata = Path(in_data).stem
    mdat = mdata.split('_')[:-1]
    mdt = '_'.join(mdat)
    # Read in the modified data
    df1 = pd.read_csv(in_data)
    # Make unique list of objects
    flist = sorted(df1['tile_id'].unique().tolist())
    completed = []
    # Loop through unique list of images (jpx)
    for item in flist:
        if item not in completed:
            completed.append(item)
        else:
            continue
        # Grab relevant data from original dataframe
        df_temp = df1.loc[df1['tile_id'] == item].copy()
        # And loop through to create new file
        for index, row in df_temp.iterrows():
            if 'X' == row.coord_label:
                x_coord = row.coord_value
                x_coord = '{:0>6d}'.format(x_coord)
                x_val = 'x' + x_coord
            if 'Y' == row.coord_label:
                y_coord = row.coord_value
                y_coord = '{:0>6d}'.format(y_coord)
                y_val = 'y' + y_coord
        curr = os.getcwd()
        old_id = item + '.jpx'
        original_path = os.path.join(curr, old_id)
        mdata = old_id.split('_')
        mdat = mdata[:-1]
        sample = mdat[0]
        series = mdat[1]
        z_coord = mdat[2]
        block = mdat[3]
        side = mdat[4]
        new_id = '_'.join([sample, side, series, block, z_coord, y_val, x_val])
        # Output new dataframe
        nd = {}
        nd['original_tile_id'] = item
        nd['original_tile_path'] = original_path
        nd['new_global_id'] = new_id
        new_df = pd.DataFrame().append(nd, ignore_index=True)
        output_path = mdat + '_AssemblyGlobal.csv'
        if os.path.exists(output_path):
            new_df.to_csv(output_path, mode='a', header=False, index=False)
        else:
            new_df.to_csv(output_path, header=True, index=False)




##############################################
##############################################
########### RUNNING THE PIPELINE #############
##############################################
##############################################




""" First, we can create an inventory of the directory to be investigated  """
# # Run the pipeline on the current directory
# make_mosaic_inventory()


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
        # This position follows the loop in which each image is transformed from wide to deep...
        # Could potentially put in a segment here to transform the final file into either .npz (zipped array) or HDF5 (???)
        deepest_array = np.load(deep_path)
        np.savez_compressed(deep_archive, deepest_array)
        # end








##############################################
##############################################



