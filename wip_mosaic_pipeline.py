# PyLearn1


Simran's edits



#21/10/06 notes merging test

Some other changes

making changes second time

third time changes
NEW LINE NEW NOTES




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

import os, os.path, pathlib, cv2
import scipy.stats
import numpy as np
import pandas as pd
import cv2, sys, math, json, glob
import os, os.path
import imageio, statistics, time
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as mpplt
import nibabel as nib
import SimpleITK as sitk
import nipype
import nipype.algorithms.metrics
import mayavi
import SimpleITK as sitk
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from skimage import color, data, exposure, filters, img_as_float, img_as_uint, io, measure
from pathlib import Path
from PIL import Image
from datetime import datetime
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import jaccard_score
from nipype.algorithms.metrics import Distance, Overlap, FuzzyOverlap, ErrorMap
from ipywidgets import interact, fixed
import png
import re, csv


##############################################
##############################################


def make_mosaic_inventory(): #creating a picture from smaller pictures
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
                tile_id = mdt + '_' + tile_num
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
##############################################


def ingest_assembly_data(in_data):
    # First, get the container metadata
    mdata = Path(in_data).stem
    medat = mdata.split('_')[:-1]
    mdt = '_'.join(medat)
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
        mdata = item.split('_')
        mdat = mdata[:-1]
        sample = mdat[0]
        series = mdat[1]
        z_coord = mdat[2]
        block = mdat[3]
        side = mdat[4]
        new_id = '_'.join([sample, side, series, block, z_coord, y_val, x_val])
        # Output new dataframe
        nd = {}
        nd['tile_id'] = item
        nd['original_path'] = original_path
        nd['new_global_id'] = new_id
        new_df = pd.DataFrame().append(nd, ignore_index=True)
        output_path = mdt + '_AssemblyGlobal.csv'
        if os.path.exists(output_path):
            new_df.to_csv(output_path, mode='a', header=False, index=False)
        else:
            new_df.to_csv(output_path, header=True, index=False)


##############################################
##############################################


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


def object_detection(image):
    curr_dir = os.getcwd()
    # For segmenting stained (black) neurons in light brown tissue -- MGN project
    #im_rgb = imageio.imread('W312_SeriesBDAd_Section0020_Brain_Whole_000210_xa1000xb1400ya0400yb)
    im_rgb = imageio.imread(image)
    metadata = Path(image).stem
    mdata = '_'.join(metadata.split('_')[:-1])
    # Convert to HSV space
    im_hsv = color.convert_colorspace(arr=im_rgb, fromspace='rgb', tospace='hsv')
    val = im_hsv[:,:,2]
    # Normalize image lighting
    claheV = exposure.equalize_adapthist(image=val, kernel_size=10, clip_limit=0.01, nbins=100)
    # Aggressively smooth image to eliminate background
    claheV = img_as_ubyte(claheV)
    bil1V = rank.mean_bilateral(image=claheV, selem=disk(radius=50), s0=100, s1=100)
    # Set up threshold parameters to eliminate outlier minimal values 
    mx = np.amax(bil1V)
    mn = np.amin(bil1V)
    rng = (mx-mn)
    thresh = mx - (rng*0.5)
    # print('max', mx, 'min', mn, 'range', rng, 'threshold', thresh)
    im = bil1V
    im[im < thresh] = 0
    # Aggressively smooth image to eliminate background
    bilm = rank.mean_bilateral(image=im, selem=disk(radius=50), s0=100, s1=100)
    # Merge nearby local minimum values, and test array range for object (!!!)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    seeds = img_as_ubyte(cv2.dilate(bilm, kernel, iterations=5))
    mx1 = np.max(seeds)
    mn1 = np.min(seeds)
    rng1 = mx1-mn1
    if rng1 > 100:
        # Set nomenclature
        out_fname = mdata + '_Object.png' 
        out_dir = '/Users/djm/Desktop/2018_python/test_targets/'
        # If object is present, write contours to file
        kernel_m = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
        mask = img_as_ubyte(cv2.dilate(bilm, kernel_m, iterations=1))
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        prediction = cv2.drawContours(mask, contours, -1, (255, 0, 0), cv2.FILLED)
        pred = invert(prediction)
        im0 = Image.fromarray(pred)
        im0.save(out_fname)
        # Now, move both files to folder containing target segmentations
        shutil.move(src=out_fname, dst=out_dir)
        shutil.move(src=image, dst=out_dir)
    else:
        # # WIP ...

        # Set nomenclature
        out_fname = mdata + '_Blank.png'
        out_dir = '/Users/djm/Desktop/2018_python/test_blanks/'
        
        
        
        ## REVISED THIS TO BE A LARGER PATCH
        mask = np.zeros([400,400], dtype=np.uint32)
        im0 = Image.fromarray(mask)
        im0.save(out_fname)
        # Now, move both files to folder containing blank segmentations
        shutil.move(src=out_fname, dst=out_dir)
        shutil.move(src=image, dst=out_dir)


####################################
####################################
####################################
###### Start Object Chunking #######
####################################

# Then read-in the output file
run_location = os.getcwd().split('/')[-1:][0]
search0 = run_location + '_InventoryMosaic.csv'
# Read in the prepared datafile containing an inventory of the mosaic files to be processed
# [df1 has 'container' , 'fname' , 'fpath']
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
    #""" Second, read-in the transformed data for each container, and loop through images """
    search2 = item + '_AssemblyGlobal.csv'
    df2 = pd.read_csv(search2)
    ln = len(df2)
    bar = IncrementalBar('Processing:', max=ln)
    # Loop through the images associated with each datafile
    for index, row in df2.iterrows():
        # For now, when testing, simply write out the file order (do not process files)
        # df2 contains: 'tile_id', 'original_path', 'new_global_id'
        tile_id = row.tile_id
        global_tile_id = row.new_global_id
        # Set up nomenclature for subarray output
        mdata = row.new_global_id.split('_')
        metadata = '_'.join(mdata[:-2])
        x_coord = int(mdata[-1][1:])
        y_coord = int(mdata[-2][1:])
        # # Read-in the file, after setting nomenclature
        # img0 = imageio.imread(fname)
        # # Get dim
        # xdim = img0.shape[1]
        # ydim = img0.shape[0]
        # When testing, simply set the dim
        xdim = 2752
        ydim = 2192
        # Derive limits
        xlim = round(xdim/400) + 1
        ylim = round(ydim/400) + 1
        # Derive cut-off value (i.e. penultimate subarray)
        xnum = xlim - 1
        ynum = ylim - 1
        # Loop through range of Y values
        # docs for fnx -- range(start, stop, interval)
        for x in range(0, xlim, 1):
            # Define cropping rectangle
            # where (x0 = left, x1 = right; y0 = upper, y1 = lower)
            # Recall: Global coordinate embedded as nomenclature must reflect x|y_coord(s)
            # First option
            if x == 0:
                left = 0
                Global_Left = left + x_coord
                right = 400
                for y in range(0, ylim, 1):
                    if y == 0:
                        upper = 0
                        Global_Upper = upper + y_coord
                        lower = 400
                        # Set coordinate values to being larger bc of global coordinate space
                        GlobalX = 'x' + '{:0>6d}'.format(Global_Left)
                        GlobalY = 'y' + '{:0>6d}'.format(Global_Upper)
                        out_fname = metadata + '_' + GlobalY + '_' + GlobalX
                        # Set up naming for output
                        nd = {}
                        nd['xdim'] = x 
                        nd['ydim'] = y 
                        nd['global_container'] = metadata
                        nd['global_tile_id'] = global_tile_id
                        nd['global_patch_id'] = out_fname
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path = metadata + '_GlobalCoordinates.csv'
                        if os.path.exists(output_path):
                            new_df.to_csv(output_path, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path, header=True, index=False)
                        
                        # WIP ......
                        #### Having successfully sketched out the subarrays to create and file to save as a record,
                        #### next step is to chunk the image itself into a deep array from a wide array (i.e. image)...
                        
                        # First, load each array
                        t_arr = np.load(row.opath)
                        t2_arr = np.load(npy_outname)
                        # Then aggregate the arrays [y,x,z] with dstack (along dim [2] as tuple)
                        t_agg = np.dstack([t_arr, t2_arr])
                        # Begin conversion to NIFTI, rotate and then extract as image with spacing
                        vol0 = t_agg.T
                        vol1 = sitk.GetImageFromArray(vol0)
                        vol1.SetSpacing([1.0,1.0,40.0])
                        # Nomenclature
                        nifti_out = some_name_eh + '.nii.gz'
                        out_npath = os.path.join(cdir, nifti_out)
                        sitk.WriteImage(vol1, nifti_out)


                        # # Create subarray 1 
                        # subarray = img0[upper:lower, left:right]
                        # Run object detection & move algorithm
                        # object_detection(subarray)

                        # # Alternative way, to write file first, and then run seg
                        # im1 = Image.fromarray(subarray)
                        # im1.save(out_fname)
                        # Run object detection & move algorithm
                        # object_detection(out_fname)
                    elif (y > 0 and y < ynum):
                        upper = y*400
                        Global_Upper = upper + y_coord
                        lower = upper + 400
                        # Set coordinate values to being larger bc of global coordinate space
                        GlobalX = 'x' + '{:0>6d}'.format(Global_Left)
                        GlobalY = 'y' + '{:0>6d}'.format(Global_Upper)
                        out_fname = metadata + '_' + GlobalY + '_' + GlobalX
                        # Set up naming for output
                        nd = {}
                        nd['xdim'] = x 
                        nd['ydim'] = y 
                        nd['global_container'] = metadata
                        nd['global_tile_id'] = global_tile_id
                        nd['global_patch_id'] = out_fname
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path = metadata + '_GlobalCoordinates.csv'
                        if os.path.exists(output_path):
                            new_df.to_csv(output_path, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path, header=True, index=False)
                        # # Create subarray 2
                        # subarray = img0[upper:lower, left:right]
                        # im1 = Image.fromarray(subarray)
                        # im1.save(out_fname)
                        # # Run object detection & move algorithm
                        # object_detection(out_fname)
                    elif y == ynum:
                        upper = 1792
                        Global_Upper = upper + y_coord
                        lower = 2192
                        # Set coordinate values to being larger bc of global coordinate space
                        GlobalX = 'x' + '{:0>6d}'.format(Global_Left)
                        GlobalY = 'y' + '{:0>6d}'.format(Global_Upper)
                        out_fname = metadata + '_' + GlobalY + '_' + GlobalX
                        # Set up naming for output
                        nd = {}
                        nd['xdim'] = x 
                        nd['ydim'] = y 
                        nd['global_container'] = metadata
                        nd['global_tile_id'] = global_tile_id
                        nd['global_patch_id'] = out_fname
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path = metadata + '_GlobalCoordinates.csv'
                        if os.path.exists(output_path):
                            new_df.to_csv(output_path, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path, header=True, index=False)
                        # # Create subarray 3
                        # subarray = img0[upper:lower, left:right]
                        # im1 = Image.fromarray(subarray)
                        # im1.save(out_fname)
                        # # Run object detection & move algorithm
                        # object_detection(out_fname)
                    else:
                        continue
            # Second option
            elif (x > 0 and x < xnum):
                left = x*400
                Global_Left = left + x_coord
                right = left + 400
                for y in range(0, ylim, 1):
                    if y == 0:
                        upper = 0
                        Global_Upper = upper + y_coord
                        lower = 400
                        # Set coordinate values to being larger bc of global coordinate space
                        GlobalX = 'x' + '{:0>6d}'.format(Global_Left)
                        GlobalY = 'y' + '{:0>6d}'.format(Global_Upper)
                        out_fname = metadata + '_' + GlobalY + '_' + GlobalX
                        # Set up naming for output
                        nd = {}
                        nd['xdim'] = x 
                        nd['ydim'] = y 
                        nd['global_container'] = metadata
                        nd['global_tile_id'] = global_tile_id
                        nd['global_patch_id'] = out_fname
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path = metadata + '_GlobalCoordinates.csv'
                        if os.path.exists(output_path):
                            new_df.to_csv(output_path, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path, header=True, index=False)
                        # # Create subarray 4
                        # subarray = img0[upper:lower, left:right]
                        # im1 = Image.fromarray(subarray)
                        # im1.save(out_fname)
                        # # Run object detection & move algorithm
                        # object_detection(out_fname)
                    elif (y > 0 and y < ynum):
                        upper = y*400
                        Global_Upper = upper + y_coord
                        lower = upper + 400
                        # Set coordinate values to being larger bc of global coordinate space
                        GlobalX = 'x' + '{:0>6d}'.format(Global_Left)
                        GlobalY = 'y' + '{:0>6d}'.format(Global_Upper)
                        out_fname = metadata + '_' + GlobalY + '_' + GlobalX
                        # Set up naming for output
                        nd = {}
                        nd['xdim'] = x 
                        nd['ydim'] = y 
                        nd['global_container'] = metadata
                        nd['global_tile_id'] = global_tile_id
                        nd['global_patch_id'] = out_fname
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path = metadata + '_GlobalCoordinates.csv'
                        if os.path.exists(output_path):
                            new_df.to_csv(output_path, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path, header=True, index=False)
                        # # Create subarray 5
                        # subarray = img0[upper:lower, left:right]
                        # im1 = Image.fromarray(subarray)
                        # im1.save(out_fname)
                        # # Run object detection & move algorithm
                        # object_detection(out_fname)
                    elif y == ynum:
                        upper = 1792
                        Global_Upper = upper + y_coord
                        lower = 2192
                        # Set coordinate values to being larger bc of global coordinate space
                        GlobalX = 'x' + '{:0>6d}'.format(Global_Left)
                        GlobalY = 'y' + '{:0>6d}'.format(Global_Upper)
                        out_fname = metadata + '_' + GlobalY + '_' + GlobalX
                        # Set up naming for output
                        nd = {}
                        nd['xdim'] = x 
                        nd['ydim'] = y 
                        nd['global_container'] = metadata
                        nd['global_tile_id'] = global_tile_id
                        nd['global_patch_id'] = out_fname
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path = metadata + '_GlobalCoordinates.csv'
                        if os.path.exists(output_path):
                            new_df.to_csv(output_path, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path, header=True, index=False)
                        # # Create subarray 6
                        # subarray = img0[upper:lower, left:right]
                        # im1 = Image.fromarray(subarray)
                        # im1.save(out_fname)
                        # # Run object detection & move algorithm
                        # object_detection(out_fname)
                    else:
                        continue
            # Third option
            elif x == xnum:
                left = 2352
                Global_Left = left + x_coord
                right = 2752
                for y in range(0, ylim, 1):
                    if y == 0:
                        upper = 0
                        Global_Upper = upper + y_coord
                        lower = 400
                        # Set coordinate values to being larger bc of global coordinate space
                        GlobalX = 'x' + '{:0>6d}'.format(Global_Left)
                        GlobalY = 'y' + '{:0>6d}'.format(Global_Upper)
                        out_fname = metadata + '_' + GlobalY + '_' + GlobalX
                        # Set up naming for output
                        nd = {}
                        nd['xdim'] = x 
                        nd['ydim'] = y 
                        nd['global_container'] = metadata
                        nd['global_tile_id'] = global_tile_id
                        nd['global_patch_id'] = out_fname
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path = metadata + '_GlobalCoordinates.csv'
                        if os.path.exists(output_path):
                            new_df.to_csv(output_path, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path, header=True, index=False)
                        # # Create subarray 7
                        # subarray = img0[upper:lower, left:right]
                        # im1 = Image.fromarray(subarray)
                        # im1.save(out_fname)
                        # # Run object detection & move algorithm
                        # object_detection(out_fname)
                    elif (y > 0 and y < ynum):
                        upper = y*400
                        Global_Upper = upper + y_coord
                        lower = upper + 400
                        # Set coordinate values to being larger bc of global coordinate space
                        GlobalX = 'x' + '{:0>6d}'.format(Global_Left)
                        GlobalY = 'y' + '{:0>6d}'.format(Global_Upper)
                        out_fname = metadata + '_' + GlobalY + '_' + GlobalX
                        # Set up naming for output
                        nd = {}
                        nd['xdim'] = x 
                        nd['ydim'] = y 
                        nd['global_container'] = metadata
                        nd['global_tile_id'] = global_tile_id
                        nd['global_patch_id'] = out_fname
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path = metadata + '_GlobalCoordinates.csv'
                        if os.path.exists(output_path):
                            new_df.to_csv(output_path, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path, header=True, index=False)
                        # # Create subarray 8
                        # subarray = img0[upper:lower, left:right]
                        # im1 = Image.fromarray(subarray)
                        # im1.save(out_fname)
                        # # Run object detection & move algorithm
                        # object_detection(out_fname)
                    elif y == ynum:
                        upper = 1792
                        Global_Upper = upper + y_coord
                        lower = 2192
                        # Set coordinate values to being larger bc of global coordinate space
                        GlobalX = 'x' + '{:0>6d}'.format(Global_Left)
                        GlobalY = 'y' + '{:0>6d}'.format(Global_Upper)
                        out_fname = metadata + '_' + GlobalY + '_' + GlobalX
                        # Set up naming for output
                        nd = {}
                        nd['xdim'] = x 
                        nd['ydim'] = y 
                        nd['global_container'] = metadata
                        nd['global_tile_id'] = global_tile_id
                        nd['global_patch_id'] = out_fname
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path = metadata + '_GlobalCoordinates.csv'
                        if os.path.exists(output_path):
                            new_df.to_csv(output_path, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path, header=True, index=False)
                        # # Create subarray 9
                        # subarray = img0[upper:lower, left:right]
                        # im1 = Image.fromarray(subarray)
                        # im1.save(out_fname)
                        # # Run object detection & move algorithm
                        # object_detection(out_fname)
                    else:
                        continue
            else:
                continue
        bar.next()
    bar.finish()
