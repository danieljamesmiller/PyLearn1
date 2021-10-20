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

def make_inventory_forMosaic():
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
                # For now, exclude the SeriesBDAd datasets....
                if 'SeriesBDAd' in name:
                    continue
                # Also exclude hidden files
                if '._' in name:
                    continue
                # and specific file types
                if '_CompilerResults' in name:
                    continue
                if '_AssemblyComplete' in name:
                    continue
                # Selectively process BDAa series.....
                if 'SeriesBDAa' in name:
                    mdata = Path(name).stem
                    item_path = os.path.join(dirpath, name)
                    container = '_'.join(mdata.split('_')[:-1])
                    # Nomenclature for output
                    nd = {}
                    nd['container'] = container
                    nd['fpath'] = item_path
                    nd['tile_id'] = name
                    # Make new dataframe with output
                    df1 = pd.DataFrame().append(nd, ignore_index=True)
                    out_fname = run_location + '_InventoryMosaic_BigPatchValidation.csv'
                    if os.path.exists(out_fname):
                        df1.to_csv(out_fname, mode='a', header=False, index=False)
                    else:
                        df1.to_csv(out_fname, header=True, index=False)
            bar.next()
        bar.finish()


##############################################
##############################################

def assembly_txt_conversion(input_file):
    """ Convert Assembly Data File Type """
    # Convert assembly.txt to assembly.csv 
    # The goal here is to make accessible all the data from the Assembly file
    core_id = Path(input_file).stem
    odir = os.getcwd() + '/BigPatchValidation'
    core_id = '_'.join(core_id.split('_')[:-1])
    txt_file = input_file
    csv_file = os.path.join(odir, (core_id + '_AssemblyConverted.csv'))
    with open(txt_file, 'rt') as infile, open(csv_file, 'w+') as outfile:
        stripped = (line.strip() for line in infile)
        lines = (line.split(",") for line in stripped if line)
        writer = csv.writer(outfile)
        writer.writerows(lines)

##############################################

def reorder_assembly_data(input_data):
    """ Ingest the Assembly Data """
    cdir = os.getcwd()
    odir = cdir + '/BigPatchValidation'
    # First, get the container metadata
    # Now we are reading in the file's path
    mdata = input_data.split('/')
    # Split on the folders, grab the final one
    fdata = mdata[-1]
    fdata_path = os.path.join(odir, fdata)
    container = Path(fdata).stem
    container = '_'.join(fdata.split('_')[:-1])
    # Split the original into parts (and ignore the end) 
    # Read in the converted original file
    df0 = pd.read_csv(fdata_path, header=None)
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
                tile_id = container + '_' + tile_num + '.jpx'
                # Set up naming for output:
                nd = {}
                nd['tile_id'] = tile_id
                nd['coord_label'] = coord_label
                nd['coord_value'] = coord_val
                new_df = pd.DataFrame().append(nd, ignore_index=True)
                output_name = container + '_AssemblyModified.csv'
                output_path = os.path.join(odir, output_name)
                if os.path.exists(output_path):
                    new_df.to_csv(output_path, mode='a', header=False, index=False)
                else:
                    new_df.to_csv(output_path, header=True, index=False)

##############################################

def ingest_assembly_data(in_data):
    # First, get the container metadata
    cdir = os.getcwd()
    odir = cdir + '/BigPatchValidation'
    # # Metadata now are dealing with an incoming Pathlike object (i.e. in_data is a path)
    # mdata = Path(in_data).stem
    # mdat = mdata.split('_')[:-1]
    # mdt = '_'.join(mdat)

    # Read in the inventory file from 
    df0 = pd.read_csv('preprocessing_InventoryMosaic_BigPatchValidation.csv')
    # Read in the '_AssemblyModified.csv' data
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
        # Set up second temporary df to effectively join these two
        # in order to get the path for the original files.
        df_t2 = df0.loc[df0['tile_id']==item].copy()
        # Create the original path....
        original_path = df_t2['fpath'].values[0]
        mdata = item.split('_')
        mdat = mdata[:-1]
        mdt = '_'.join(mdat)
        sample = mdat[0]
        series = mdat[1]
        z_coord = mdat[2]
        block = mdat[3]
        side = mdat[4]
        new_id = '_'.join([sample, side, series, block, z_coord, y_val, x_val])
        # Output new dataframe
        nd = {}
        nd['original_tile_id'] = item
        nd['global_tile_id'] = mdt
        nd['global_patch_id'] = new_id
        nd['original_path'] = original_path
        new_df = pd.DataFrame().append(nd, ignore_index=True)
        output_fname = mdt + '_AssemblyGlobal.csv'
        output_path = os.path.join(odir, output_fname)
        if os.path.exists(output_path):
            new_df.to_csv(output_path, mode='a', header=False, index=False)
        else:
            new_df.to_csv(output_path, header=True, index=False)


##############################################

""" 
RUN THE CHUNKING PIPELINE ON DIRECTORY 
"""
# Establish nomenclature
cdir = os.getcwd()
run_location = os.getcwd().split('/')[-1:][0]
# Establish output directory
odir = cdir + '/BigPatchValidation'
# Read in the prepared datafile containing an inventory of the mosaic files to be processed
# Imported df [has 'container' , 'tile_id' , 'fpath']
df1 = pd.read_csv('preprocessing_InventoryMosaic_BigPatchValidation.csv')
# make a list of all the unique containers
container_list = sorted(df1['container'].unique().tolist())
ln = len(container_list)
bar = IncrementalBar('Processing:', max=ln)
""" Loop through the containers to get global coordinate nomenclature """
completed = []
for item in container_list:
    if item not in completed:
        completed.append(item)
        # First, get the Assembly Data file and process it...
        search1 = item + '_AssemblyData.txt'
        # Get file location from prepared file
        df_temp = df1.loc[df1['tile_id'] == search1].copy()
        # Ignore empty frames
        if len(df_temp) < 1:
            continue
        # Run first function
        # Get first file path for container mosaic data
        assemble_fpath = df_temp['fpath'].values[0]
        assembly_txt_conversion(assemble_fpath)
        # Next file path and function
        reorder_fname = item + '_AssemblyConverted.csv'
        reorder_fpath = os.path.join(odir, reorder_fname)
        reorder_assembly_data(reorder_fpath)
        # Final file path and function
        ingest_fname = item + '_AssemblyModified.csv'
        ingest_fpath = os.path.join(odir, ingest_fname)
        ingest_assembly_data(ingest_fpath)
        """ 
        Second, read-in the transformed data for each container, and loop through images 
        """
        search_2 = item + '_AssemblyGlobal.csv'
        search2 = os.path.join(odir, search_2)
        df2 = pd.read_csv(search2)
        # Loop through the images associated with each container's datafile
        for index, row in df2.iterrows():
            # For now, when testing, simply write out the file order (do not process files)
            # df2 contains: 'original_tile_id', 'original_path', 'new_global_id' of each original tile image
            fname = row.original_path
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
            csv_outname = metadata + '_GlobalCoordinates.csv'
            csv_outpath = os.path.join(odir, csv_outname)
            deep_patch = global_tile_id + '.npy'
            patch_path = os.path.join(odir, deep_patch)
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
                            np.save(patch_path, patch)
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
                            # Check that the deep array is present to append
                            if os.path.exists(patch_path):
                                deep_arr = np.load(patch_path)
                                deeper_arr = np.dstack([deep_arr, patch])
                                # Write deeper_arr to file
                                np.save(patch_path, deeper_arr)
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
                            if os.path.exists(patch_path):
                                deep_arr = np.load(patch_path)
                                deeper_arr = np.dstack([deep_arr, patch])
                                # Write deeper_arr to file
                                np.save(patch_path, deeper_arr)
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
                            if os.path.exists(patch_path):
                                deep_arr = np.load(patch_path)
                                deeper_arr = np.dstack([deep_arr, patch])
                                # Write deeper_arr to file
                                np.save(patch_path, deeper_arr)
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
                            if os.path.exists(patch_path):
                                deep_arr = np.load(patch_path)
                                deeper_arr = np.dstack([deep_arr, patch])
                                # Write deeper_arr to file
                                np.save(patch_path, deeper_arr)
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
                            if os.path.exists(patch_path):
                                deep_arr = np.load(patch_path)
                                deeper_arr = np.dstack([deep_arr, patch])
                                # Write deeper_arr to file
                                np.save(patch_path, deeper_arr)
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
                            if os.path.exists(patch_path):
                                deep_arr = np.load(patch_path)
                                deeper_arr = np.dstack([deep_arr, patch])
                                # Write deeper_arr to file
                                np.save(patch_path, deeper_arr)
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
                            if os.path.exists(patch_path):
                                deep_arr = np.load(patch_path)
                                deeper_arr = np.dstack([deep_arr, patch])
                                # Write deeper_arr to file
                                np.save(patch_path, deeper_arr)
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
                            if os.path.exists(patch_path):
                                deep_arr = np.load(patch_path)
                                deeper_arr = np.dstack([deep_arr, patch])
                                # Write deeper_arr to file
                                np.save(patch_path, deeper_arr)
                        # Conclusion of y values range
                        else:
                            continue
                # Should be the conclusion of the x values range
                else:
                    continue
            # This position follows the loop in which each image is transformed from wide to deep...
            # Could potentially put in a segment here to transform the final file into either .npz (zipped array) or HDF5 (???)
            patches_compressed = global_tile_id + '.npz'
            patches_path = os.path.join(odir, patches_compressed)
            deepest_array = np.load(patch_path)
            np.savez_compressed(patches_path, deepest_array)
            # Finally, clean up the directory by removing temporary files (i.e. all the npy)
            os.remove(patch_path)
    bar.next()
bar.finish()
        




##############################################

##############################################

# Script to process images in the MGN project
# for use as a first novel_test_set of the BigPatches
# nnunet model (model #5), and subsequent validation 
# by user annotation in Anatolution.

# Potentially run the mosaic pipeline on all the BDAa files from W310?
# Run this from ~/Volumes/EIS3/DL0/deep_learning/preprocessing
# Inside /preprocessing are dirs for BDAa and BDAd series



