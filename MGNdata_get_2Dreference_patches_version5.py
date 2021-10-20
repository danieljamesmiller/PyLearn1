

""" Module to output training data (MGN project) as subarrays of each objects + surround [HSV] for 2D data """


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
from progress.bar import IncrementalBar


# # Read-in the image catalogue....
# df_microscopy = pd.read_csv('BB0_microscopy_archive_25-11-2020.csv')
df_microscopy = pd.read_csv('MGN_project_ImageCatalogue_08102021.csv')
print(datetime.now())
# Read-in the patch_polygon data inventory...
df_dat = pd.read_csv('project_MGNfelis_Oct2021_PatchesTraining.csv')
ln = len(df_dat)
bar = IncrementalBar('Processing:', max=ln)
cdir = os.getcwd()
odir = cdir + '/MGNpatches400_refs'
# Loop through the path_polygon data to find ref_subarrays...
for index, row in df_dat.iterrows():
    # Derive metadata of interest
    metadata = Path(row.mask_id).stem
    mdata = metadata.split('_')
    new_mdata = '_'.join(mdata[0:8])
    new_name = new_mdata + '_Reference.png'
    x0 = round(row.x0)
    x1 = round(row.x1)
    y0 = round(row.y0)
    y1 = round(row.y1)
    # Load the reference image from other dataframe
    df_temp = df_microscopy.loc[df_microscopy['image_id'] == row.ref_key]
    for i, r in df_temp.iterrows():
        # Read-in the file
        new_image = imageio.imread(r.image_path)
        # First, subselect the array to be saved
        # These are still RGB = depth of 3???
        # May need to flip the orientation of the axes around???
        subarray = new_image[y0:y1, x0:x1, :]
        # Second, assign the subarray as a Pillow Image object
        im = Image.fromarray(subarray)
        # Assign the path for the output
        out_npath = os.path.join(odir, new_name)
        # Call the save image function
        im.save(out_npath)
        # Save file names and locations to csv as inventory/catalogue
        nd = {}
        # Note: r.pulls from df_microscopy | row.pulls from df_dat
        nd['original_image'] = r.image_id
        nd['or_im_path'] = r.image_path
        nd['original_mask'] = row.mask_id
        nd['or_mk_path'] = row.mask_path
        nd['object_type'] = row.ob_class
        nd['user'] = row.ob_user
        nd['centroidX'] = row.centroidX
        nd['centroidY'] = row.centroidY
        nd['object_area'] = row.object_area
        nd['object_radius'] = row.object_radius
        nd['x0'] = x0
        nd['x1'] = x1
        nd['y0'] = y0
        nd['y1'] = y1
        # Create output CSV for downstream aggregation
        new_df = pd.DataFrame().append(nd, ignore_index=True)
        output_path_to_new_file = os.path.join(odir, 'project_MGNfelis_Oct2021_PatchesReferences.csv')
        if os.path.exists(output_path_to_new_file):
            new_df.to_csv(output_path_to_new_file, mode='a', header=False, index=False)
        else:
            new_df.to_csv(output_path_to_new_file, header=True, index=False)
    bar.next()
bar.finish()
print(datetime.now())