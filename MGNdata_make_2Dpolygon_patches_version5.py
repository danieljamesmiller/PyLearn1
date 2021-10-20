
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
import png

#################################################

def join_coords(rows):
    coords = []
    for i, r in rows.iterrows():
        row_coord = [r.x_val, r.y_val]
        coords.append(row_coord)
    return coords

#################################################

def make_patch_polygons(input_df):  
    # STEP 1: CREATE 2D DATASETS -- Create GROUP-BY object at the level of each layer in each object (for MGN, this is all we did was 2D drawings)
    cdir = os.getcwd()
    odir = cdir + '/MGNpatches400_masks'
    # Group points within each layer (i.e. transform list of polygon points into 2D polygons) 
    # The 'oid' is the variable of interest; 'oid' specifies the object layer identifier (i.e. the contour number;; cell type is common to this label dimension)
    gb0 = input_df.groupby(['container', 'user', 'otype', 'z_val', 'oid']).apply(join_coords).reset_index()
    gb0.rename(columns={0:'coordinates'}, inplace=True)
    ln = len(gb0)
    bar = IncrementalBar('Processing:', max=ln)
    # Begin loop through of the grouped-by df, ordered by contour identifier ('oid')
    for index, row in gb0.iterrows():
        container = row.container
        # Revise erroneous original nomenclature for W310 & W312: 
        # Rename to "W310_SeriesBDAd" and "W312_SeriesBDAa" 
        mdat = container.split('_')
        first = mdat[0]
        tdat = mdat[1]
        others = mdat[2:]
        if 'W310' in container:
            tdat = 'SeriesBDAd'
        if 'W312' in container:
            tdat = 'SeriesBDAa'
        # Re-form nomenclature per input
        n_container = first +'_'+ tdat +'_'+ '_'.join(others)
        user = row.user[:6]
        oid = row.oid
        otype = row.otype
        if 'Marker 1' == otype:
            otype = 'Neuron'
        if 'Marker 2' == otype:
            otype = 'Artefact'
        omdata = otype
        # ONLY process images << with the string 'BDA' >> in it for now...
        if 'BDA' in container:
            """ Calculate object centroid data """
            # Format the z-value to have 100000s place (for downstream sorting on number)
            zval = '{:0>6d}'.format(row.z_val)
            # Format the container for 2D datasets
            container_2D = n_container + '_' + zval
            # call coordinate points as array 
            pts_array = np.array(row.coordinates, dtype=np.int32)
            # create empty array for polygons [!!! check ordering of [rows=y, columns=x]]
            p_msk = np.zeros([2192,2752], dtype=np.uint8)
            # call draw polygon function on coordinates and empty array 
            p_mask = cv2.fillConvexPoly(p_msk, pts_array, 255)
            # Next, take some measurements for clustering...
            # Find bounding box to determine massive outliers for exclusion
            x,y,w,h = cv2.boundingRect(p_mask)
            # SIZE EXCLUSION CRITERION
            # Do not process objects that are too big to fit in FOV
            if h > 400:
                continue
            if w > 400:
                continue
            # print('x:', x, 'y:', y, 'w:', w, 'h:', h)
            # Get area of object for comparisons on radius
            area = cv2.contourArea(pts_array)
            # Derive the radius of a cirle from area (A = pi*r^2)
            radius = (area**(1.0/2))*(1/3.1415926)
            # and to determine subarray centroid coordinates (!!!)
            Mmnts = cv2.moments(pts_array)
            # Ensure there are no zero values in division (!!!)
            m00 = int(Mmnts['m00'])
            m10 = int(Mmnts['m10'])
            m01 = int(Mmnts['m01'])
            tarr = np.array([m00, m10, m01])
            # Check for zero values in centroid coordinate data
            if np.all((tarr == 0)):
                continue
            else:
                if m00 == 0:
                    m00 == 1
                    cx = round(m10 / m00)
                    cy = round(m01 / m00)
                if m10 == 0:
                    m10 == 1
                    if m00 == 0:
                        m00 == 1
                    cx = round(m10 / m00)
                    cy = round(m01 / m00)
                if m01 == 0:
                    m01 == 1
                    if m00 == 0:
                        m00 == 1
                    cx = round(m10 / m00)
                    cy = round(m01 / m00)
                else:
                    # Else calculate centroid coordinates as normal
                    cx = round(m10 / m00)
                    cy = round(m01 / m00)
                # print('cx -', cx, '|   cy -', cy)
                """ Set up patch generation for each layer of each object """
                # Now that we have the centroid data as cx and cy...
                # Address the center-bias problem discussed w/ Khan 4/30/21.
                # Here, X and Y values are subdivided and looped; offsets are dynamically set within image boundaries
                # Run through the x values, and then embed the y values loop
                # Edge case centroid coordinates are taken 'as-is' - no offset 
                if (cx >= 0) and (cx <= 200):
                    # Edge case centroid coordinates are taken 'as-is' - no offset for X values
                    # So just go straight to building the options for cy
                    cx0 = 0
                    cx1 = 400
                    if (cy >= 0) and (cy <= 200):
                        # No offset
                        cy0 = 0
                        cy1 = 400
                        """ OUTPUT """
                        # Create Nomenclature
                        # Format the z-value to have 100000s place (for downstream sorting on number)
                        x0_val = '{:0>6d}'.format(cx0)
                        y0_val = '{:0>6d}'.format(cy0)
                        x1_val = '{:0>6d}'.format(cx1)
                        y1_val = '{:0>6d}'.format(cy1)
                        ob_subarray = 'y' + str(y0_val) + '_' + 'x' + str(x0_val)
                        object_identity =  ob_subarray + '_' + omdata
                        filename = container_2D + '_' + object_identity + '.png'
                        # Establish the path for the image when output
                        out_npath = os.path.join(odir, filename)
                        # Select the array to be saved from data
                        subarray = p_mask[cy0:cy1, cx0:cx1]
                        # Convert the array to an image
                        im = Image.fromarray(subarray)
                        # Call the save image function
                        im.save(out_npath)
                        # Export metadata as csv for downstream processing
                        # First, get the name of the original reference file to search on...
                        # Then, record this file's identity and metadata of interest
                        nd = {}
                        nd['ref_key'] = container_2D
                        nd['mask_id'] = filename
                        nd['mask_path'] = out_npath
                        nd['ob_class'] = otype
                        nd['ob_user'] = user
                        nd['centroidX'] = cx
                        nd['centroidY'] = cy
                        nd['object_area'] = area
                        nd['object_radius'] = radius
                        nd['x0'] = x0_val
                        nd['x1'] = x1_val
                        nd['y0'] = y0_val
                        nd['y1'] = y1_val
                        # Create output CSV for downstream aggregation
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path_to_new_file = 'project_MGNfelis_Oct2021_PatchesTraining.csv'
                        if os.path.exists(output_path_to_new_file):
                            new_df.to_csv(output_path_to_new_file, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path_to_new_file, header=True, index=False)
                    elif (cy >= 201) and (cy <= 649):
                        # Offset (as positive for first half of image dim)
                        # Calculate the residual (i.e. distance from closest threshold) to dynamically set the offset
                        prev_bound = 200
                        res = cy - prev_bound
                        off_set = round(res*(0.25))
                        # Set the upper coordinate
                        cy0 = (cy - 200) + off_set
                        cy1 = (cy + 200) + off_set
                        """ OUTPUT """
                        # Create Nomenclature
                        # Format the z-value to have 100000s place (for downstream sorting on number)
                        x0_val = '{:0>6d}'.format(cx0)
                        y0_val = '{:0>6d}'.format(cy0)
                        x1_val = '{:0>6d}'.format(cx1)
                        y1_val = '{:0>6d}'.format(cy1)
                        ob_subarray = 'y' + str(y0_val) + '_' + 'x' + str(x0_val)
                        object_identity =  ob_subarray + '_' + omdata
                        filename = container_2D + '_' + object_identity + '.png'
                        # Establish the path for the image when output
                        out_npath = os.path.join(odir, filename)
                        # Select the array to be saved from data
                        subarray = p_mask[cy0:cy1, cx0:cx1]
                        # Convert the array to an image
                        im = Image.fromarray(subarray)
                        # Call the save image function
                        im.save(out_npath)
                        # Export metadata as csv for downstream processing
                        # First, get the name of the original reference file to search on...
                        # Then, record this file's identity and metadata of interest
                        nd = {}
                        nd['ref_key'] = container_2D
                        nd['mask_id'] = filename
                        nd['mask_path'] = out_npath
                        nd['ob_class'] = otype
                        nd['ob_user'] = user
                        nd['centroidX'] = cx
                        nd['centroidY'] = cy
                        nd['object_area'] = area
                        nd['object_radius'] = radius
                        nd['x0'] = x0_val
                        nd['x1'] = x1_val
                        nd['y0'] = y0_val
                        nd['y1'] = y1_val
                        # Create output CSV for downstream aggregation
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path_to_new_file = 'project_MGNfelis_Oct2021_PatchesTraining.csv'
                        if os.path.exists(output_path_to_new_file):
                            new_df.to_csv(output_path_to_new_file, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path_to_new_file, header=True, index=False)
                    elif (cy >= 650) and (cy <= 1099):
                        # Offset (as positive for first half of image dim)
                        # Calculate the residual (i.e. distance from closest threshold) to dynamically set the offset
                        prev_bound = 649
                        res = cy - prev_bound
                        off_set = round(res*(0.25))
                        # Set the upper coordinate
                        cy0 = (cy - 200) + off_set
                        cy1 = (cy + 200) + off_set
                        """ OUTPUT """
                        # Create Nomenclature
                        # Format the z-value to have 100000s place (for downstream sorting on number)
                        x0_val = '{:0>6d}'.format(cx0)
                        y0_val = '{:0>6d}'.format(cy0)
                        x1_val = '{:0>6d}'.format(cx1)
                        y1_val = '{:0>6d}'.format(cy1)
                        ob_subarray = 'y' + str(y0_val) + '_' + 'x' + str(x0_val)
                        object_identity =  ob_subarray + '_' + omdata
                        filename = container_2D + '_' + object_identity + '.png'
                        # Establish the path for the image when output
                        out_npath = os.path.join(odir, filename)
                        # Select the array to be saved from data
                        subarray = p_mask[cy0:cy1, cx0:cx1]
                        # Convert the array to an image
                        im = Image.fromarray(subarray)
                        # Call the save image function
                        im.save(out_npath)
                        # Export metadata as csv for downstream processing
                        # First, get the name of the original reference file to search on...
                        # Then, record this file's identity and metadata of interest
                        nd = {}
                        nd['ref_key'] = container_2D
                        nd['mask_id'] = filename
                        nd['mask_path'] = out_npath
                        nd['ob_class'] = otype
                        nd['ob_user'] = user
                        nd['centroidX'] = cx
                        nd['centroidY'] = cy
                        nd['object_area'] = area
                        nd['object_radius'] = radius
                        nd['x0'] = x0_val
                        nd['x1'] = x1_val
                        nd['y0'] = y0_val
                        nd['y1'] = y1_val
                        # Create output CSV for downstream aggregation
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path_to_new_file = 'project_MGNfelis_Oct2021_PatchesTraining.csv'
                        if os.path.exists(output_path_to_new_file):
                            new_df.to_csv(output_path_to_new_file, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path_to_new_file, header=True, index=False)
                    elif (cy >= 1100) and (cy <= 1548):
                        # Offset (as negative for second half of image dim)
                        # Calculate the residual (i.e. distance from closest threshold) to dynamically set the offset
                        prev_bound = 1099
                        res = cy - prev_bound
                        off_set = round(res*(0.25))
                        # Set the upper coordinate
                        cy0 = (cy - 200) - off_set
                        cy1 = (cy + 200) - off_set
                        """ OUTPUT """
                        # Create Nomenclature
                        # Format the z-value to have 100000s place (for downstream sorting on number)
                        x0_val = '{:0>6d}'.format(cx0)
                        y0_val = '{:0>6d}'.format(cy0)
                        x1_val = '{:0>6d}'.format(cx1)
                        y1_val = '{:0>6d}'.format(cy1)
                        ob_subarray = 'y' + str(y0_val) + '_' + 'x' + str(x0_val)
                        object_identity =  ob_subarray + '_' + omdata
                        filename = container_2D + '_' + object_identity + '.png'
                        # Establish the path for the image when output
                        out_npath = os.path.join(odir, filename)
                        # Select the array to be saved from data
                        subarray = p_mask[cy0:cy1, cx0:cx1]
                        # Convert the array to an image
                        im = Image.fromarray(subarray)
                        # Call the save image function
                        im.save(out_npath)
                        # Export metadata as csv for downstream processing
                        # First, get the name of the original reference file to search on...
                        # Then, record this file's identity and metadata of interest
                        nd = {}
                        nd['ref_key'] = container_2D
                        nd['mask_id'] = filename
                        nd['mask_path'] = out_npath
                        nd['ob_class'] = otype
                        nd['ob_user'] = user
                        nd['centroidX'] = cx
                        nd['centroidY'] = cy
                        nd['object_area'] = area
                        nd['object_radius'] = radius
                        nd['x0'] = x0_val
                        nd['x1'] = x1_val
                        nd['y0'] = y0_val
                        nd['y1'] = y1_val
                        # Create output CSV for downstream aggregation
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path_to_new_file = 'project_MGNfelis_Oct2021_PatchesTraining.csv'
                        if os.path.exists(output_path_to_new_file):
                            new_df.to_csv(output_path_to_new_file, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path_to_new_file, header=True, index=False)
                    elif (cy >= 1549) and (cy <= 1991):
                        # Offset (as negative for second half of image dim)
                        # Calculate the residual (i.e. distance from closest threshold) to dynamically set the offset
                        prev_bound = 1548
                        res = cy - prev_bound
                        off_set = round(res*(0.25))
                        # Set the upper coordinate
                        cy0 = (cy - 200) - off_set
                        cy1 = (cy + 200) - off_set
                        """ OUTPUT """
                        # Create Nomenclature
                        # Format the z-value to have 100000s place (for downstream sorting on number)
                        x0_val = '{:0>6d}'.format(cx0)
                        y0_val = '{:0>6d}'.format(cy0)
                        x1_val = '{:0>6d}'.format(cx1)
                        y1_val = '{:0>6d}'.format(cy1)
                        ob_subarray = 'y' + str(y0_val) + '_' + 'x' + str(x0_val)
                        object_identity =  ob_subarray + '_' + omdata
                        filename = container_2D + '_' + object_identity + '.png'
                        # Establish the path for the image when output
                        out_npath = os.path.join(odir, filename)
                        # Select the array to be saved from data
                        subarray = p_mask[cy0:cy1, cx0:cx1]
                        # Convert the array to an image
                        im = Image.fromarray(subarray)
                        # Call the save image function
                        im.save(out_npath)
                        # Export metadata as csv for downstream processing
                        # First, get the name of the original reference file to search on...
                        # Then, record this file's identity and metadata of interest
                        nd = {}
                        nd['ref_key'] = container_2D
                        nd['mask_id'] = filename
                        nd['mask_path'] = out_npath
                        nd['ob_class'] = otype
                        nd['ob_user'] = user
                        nd['centroidX'] = cx
                        nd['centroidY'] = cy
                        nd['object_area'] = area
                        nd['object_radius'] = radius
                        nd['x0'] = x0_val
                        nd['x1'] = x1_val
                        nd['y0'] = y0_val
                        nd['y1'] = y1_val
                        # Create output CSV for downstream aggregation
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path_to_new_file = 'project_MGNfelis_Oct2021_PatchesTraining.csv'
                        if os.path.exists(output_path_to_new_file):
                            new_df.to_csv(output_path_to_new_file, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path_to_new_file, header=True, index=False)
                    elif (cy >= 1992) and (cy <= 2192):
                        # No offset
                        cy0 = 1792
                        cy1 = 2192
                        """ OUTPUT """
                        # Create Nomenclature
                        # Format the z-value to have 100000s place (for downstream sorting on number)
                        x0_val = '{:0>6d}'.format(cx0)
                        y0_val = '{:0>6d}'.format(cy0)
                        x1_val = '{:0>6d}'.format(cx1)
                        y1_val = '{:0>6d}'.format(cy1)
                        ob_subarray = 'y' + str(y0_val) + '_' + 'x' + str(x0_val)
                        object_identity =  ob_subarray + '_' + omdata
                        filename = container_2D + '_' + object_identity + '.png'
                        # Establish the path for the image when output
                        out_npath = os.path.join(odir, filename)
                        # Select the array to be saved from data
                        subarray = p_mask[cy0:cy1, cx0:cx1]
                        # Convert the array to an image
                        im = Image.fromarray(subarray)
                        # Call the save image function
                        im.save(out_npath)
                        # Export metadata as csv for downstream processing
                        # First, get the name of the original reference file to search on...
                        # Then, record this file's identity and metadata of interest
                        nd = {}
                        nd['ref_key'] = container_2D
                        nd['mask_id'] = filename
                        nd['mask_path'] = out_npath
                        nd['ob_class'] = otype
                        nd['ob_user'] = user
                        nd['centroidX'] = cx
                        nd['centroidY'] = cy
                        nd['object_area'] = area
                        nd['object_radius'] = radius
                        nd['x0'] = x0_val
                        nd['x1'] = x1_val
                        nd['y0'] = y0_val
                        nd['y1'] = y1_val
                        # Create output CSV for downstream aggregation
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path_to_new_file = 'project_MGNfelis_Oct2021_PatchesTraining.csv'
                        if os.path.exists(output_path_to_new_file):
                            new_df.to_csv(output_path_to_new_file, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path_to_new_file, header=True, index=False)
                    else:
                        print('wtf!!!', container, oid)
                        continue
                # Now set up patch generation for interior coordinates
                elif (cx >= 201) and (cx <= 789):
                    # Set up conditional for x values
                    # Offset (as positive for first half of image dim)
                    # Calculate the residual (i.e. distance from closest threshold) to dynamically set the offset
                    prev_bound = 200
                    res = cx - prev_bound
                    off_set = round(res*(0.25))
                    # Set the upper coordinate
                    cx0 = (cx - 200) + off_set
                    cx1 = (cx + 200) + off_set
                    if (cy >= 0) and (cy <= 200):
                        # No offset
                        cy0 = 0
                        cy1 = 400
                        """ OUTPUT """
                        # Create Nomenclature
                        # Format the z-value to have 100000s place (for downstream sorting on number)
                        x0_val = '{:0>6d}'.format(cx0)
                        y0_val = '{:0>6d}'.format(cy0)
                        x1_val = '{:0>6d}'.format(cx1)
                        y1_val = '{:0>6d}'.format(cy1)
                        ob_subarray = 'y' + str(y0_val) + '_' + 'x' + str(x0_val)
                        object_identity =  ob_subarray + '_' + omdata
                        filename = container_2D + '_' + object_identity + '.png'
                        # Establish the path for the image when output
                        out_npath = os.path.join(odir, filename)
                        # Select the array to be saved from data
                        subarray = p_mask[cy0:cy1, cx0:cx1]
                        # Convert the array to an image
                        im = Image.fromarray(subarray)
                        # Call the save image function
                        im.save(out_npath)
                        # Export metadata as csv for downstream processing
                        # First, get the name of the original reference file to search on...
                        # Then, record this file's identity and metadata of interest
                        nd = {}
                        nd['ref_key'] = container_2D
                        nd['mask_id'] = filename
                        nd['mask_path'] = out_npath
                        nd['ob_class'] = otype
                        nd['ob_user'] = user
                        nd['centroidX'] = cx
                        nd['centroidY'] = cy
                        nd['object_area'] = area
                        nd['object_radius'] = radius
                        nd['x0'] = x0_val
                        nd['x1'] = x1_val
                        nd['y0'] = y0_val
                        nd['y1'] = y1_val
                        # Create output CSV for downstream aggregation
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path_to_new_file = 'project_MGNfelis_Oct2021_PatchesTraining.csv'
                        if os.path.exists(output_path_to_new_file):
                            new_df.to_csv(output_path_to_new_file, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path_to_new_file, header=True, index=False)
                    elif (cy >= 201) and (cy <= 649):
                        # Offset (as positive for first half of image dim)
                        # Calculate the residual (i.e. distance from closest threshold) to dynamically set the offset
                        prev_bound = 200
                        res = cy - prev_bound
                        off_set = round(res*(0.25))
                        # Set the upper coordinate
                        cy0 = (cy - 200) + off_set
                        cy1 = (cy + 200) + off_set
                        """ OUTPUT """
                        # Create Nomenclature
                        # Format the z-value to have 100000s place (for downstream sorting on number)
                        x0_val = '{:0>6d}'.format(cx0)
                        y0_val = '{:0>6d}'.format(cy0)
                        x1_val = '{:0>6d}'.format(cx1)
                        y1_val = '{:0>6d}'.format(cy1)
                        ob_subarray = 'y' + str(y0_val) + '_' + 'x' + str(x0_val)
                        object_identity =  ob_subarray + '_' + omdata
                        filename = container_2D + '_' + object_identity + '.png'
                        # Establish the path for the image when output
                        out_npath = os.path.join(odir, filename)
                        # Select the array to be saved from data
                        subarray = p_mask[cy0:cy1, cx0:cx1]
                        # Convert the array to an image
                        im = Image.fromarray(subarray)
                        # Call the save image function
                        im.save(out_npath)
                        # Export metadata as csv for downstream processing
                        # First, get the name of the original reference file to search on...
                        # Then, record this file's identity and metadata of interest
                        nd = {}
                        nd['ref_key'] = container_2D
                        nd['mask_id'] = filename
                        nd['mask_path'] = out_npath
                        nd['ob_class'] = otype
                        nd['ob_user'] = user
                        nd['centroidX'] = cx
                        nd['centroidY'] = cy
                        nd['object_area'] = area
                        nd['object_radius'] = radius
                        nd['x0'] = x0_val
                        nd['x1'] = x1_val
                        nd['y0'] = y0_val
                        nd['y1'] = y1_val
                        # Create output CSV for downstream aggregation
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path_to_new_file = 'project_MGNfelis_Oct2021_PatchesTraining.csv'
                        if os.path.exists(output_path_to_new_file):
                            new_df.to_csv(output_path_to_new_file, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path_to_new_file, header=True, index=False)
                    elif (cy >= 650) and (cy <= 1099):
                        # Offset (as positive for first half of image dim)
                        # Calculate the residual (i.e. distance from closest threshold) to dynamically set the offset
                        prev_bound = 649
                        res = cy - prev_bound
                        off_set = round(res*(0.25))
                        # Set the upper coordinate
                        cy0 = (cy - 200) + off_set
                        cy1 = (cy + 200) + off_set
                        """ OUTPUT """
                        # Create Nomenclature
                        # Format the z-value to have 100000s place (for downstream sorting on number)
                        x0_val = '{:0>6d}'.format(cx0)
                        y0_val = '{:0>6d}'.format(cy0)
                        x1_val = '{:0>6d}'.format(cx1)
                        y1_val = '{:0>6d}'.format(cy1)
                        ob_subarray = 'y' + str(y0_val) + '_' + 'x' + str(x0_val)
                        object_identity =  ob_subarray + '_' + omdata
                        filename = container_2D + '_' + object_identity + '.png'
                        # Establish the path for the image when output
                        out_npath = os.path.join(odir, filename)
                        # Select the array to be saved from data
                        subarray = p_mask[cy0:cy1, cx0:cx1]
                        # Convert the array to an image
                        im = Image.fromarray(subarray)
                        # Call the save image function
                        im.save(out_npath)
                        # Export metadata as csv for downstream processing
                        # First, get the name of the original reference file to search on...
                        # Then, record this file's identity and metadata of interest
                        nd = {}
                        nd['ref_key'] = container_2D
                        nd['mask_id'] = filename
                        nd['mask_path'] = out_npath
                        nd['ob_class'] = otype
                        nd['ob_user'] = user
                        nd['centroidX'] = cx
                        nd['centroidY'] = cy
                        nd['object_area'] = area
                        nd['object_radius'] = radius
                        nd['x0'] = x0_val
                        nd['x1'] = x1_val
                        nd['y0'] = y0_val
                        nd['y1'] = y1_val
                        # Create output CSV for downstream aggregation
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path_to_new_file = 'project_MGNfelis_Oct2021_PatchesTraining.csv'
                        if os.path.exists(output_path_to_new_file):
                            new_df.to_csv(output_path_to_new_file, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path_to_new_file, header=True, index=False)
                    elif (cy >= 1100) and (cy <= 1548):
                        # Offset (as negative for second half of image dim)
                        # Calculate the residual (i.e. distance from closest threshold) to dynamically set the offset
                        prev_bound = 1099
                        res = cy - prev_bound
                        off_set = round(res*(0.25))
                        # Set the upper coordinate
                        cy0 = (cy - 200) - off_set
                        cy1 = (cy + 200) - off_set
                        """ OUTPUT """
                        # Create Nomenclature
                        # Format the z-value to have 100000s place (for downstream sorting on number)
                        x0_val = '{:0>6d}'.format(cx0)
                        y0_val = '{:0>6d}'.format(cy0)
                        x1_val = '{:0>6d}'.format(cx1)
                        y1_val = '{:0>6d}'.format(cy1)
                        ob_subarray = 'y' + str(y0_val) + '_' + 'x' + str(x0_val)
                        object_identity =  ob_subarray + '_' + omdata
                        filename = container_2D + '_' + object_identity + '.png'
                        # Establish the path for the image when output
                        out_npath = os.path.join(odir, filename)
                        # Select the array to be saved from data
                        subarray = p_mask[cy0:cy1, cx0:cx1]
                        # Convert the array to an image
                        im = Image.fromarray(subarray)
                        # Call the save image function
                        im.save(out_npath)
                        # Export metadata as csv for downstream processing
                        # First, get the name of the original reference file to search on...
                        # Then, record this file's identity and metadata of interest
                        nd = {}
                        nd['ref_key'] = container_2D
                        nd['mask_id'] = filename
                        nd['mask_path'] = out_npath
                        nd['ob_class'] = otype
                        nd['ob_user'] = user
                        nd['centroidX'] = cx
                        nd['centroidY'] = cy
                        nd['object_area'] = area
                        nd['object_radius'] = radius
                        nd['x0'] = x0_val
                        nd['x1'] = x1_val
                        nd['y0'] = y0_val
                        nd['y1'] = y1_val
                        # Create output CSV for downstream aggregation
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path_to_new_file = 'project_MGNfelis_Oct2021_PatchesTraining.csv'
                        if os.path.exists(output_path_to_new_file):
                            new_df.to_csv(output_path_to_new_file, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path_to_new_file, header=True, index=False)
                    elif (cy >= 1549) and (cy <= 1991):
                        # Offset (as negative for second half of image dim)
                        # Calculate the residual (i.e. distance from closest threshold) to dynamically set the offset
                        prev_bound = 1548
                        res = cy - prev_bound
                        off_set = round(res*(0.25))
                        # Set the upper coordinate
                        cy0 = (cy - 200) - off_set
                        cy1 = (cy + 200) - off_set
                        """ OUTPUT """
                        # Create Nomenclature
                        # Format the z-value to have 100000s place (for downstream sorting on number)
                        x0_val = '{:0>6d}'.format(cx0)
                        y0_val = '{:0>6d}'.format(cy0)
                        x1_val = '{:0>6d}'.format(cx1)
                        y1_val = '{:0>6d}'.format(cy1)
                        ob_subarray = 'y' + str(y0_val) + '_' + 'x' + str(x0_val)
                        object_identity =  ob_subarray + '_' + omdata
                        filename = container_2D + '_' + object_identity + '.png'
                        # Establish the path for the image when output
                        out_npath = os.path.join(odir, filename)
                        # Select the array to be saved from data
                        subarray = p_mask[cy0:cy1, cx0:cx1]
                        # Convert the array to an image
                        im = Image.fromarray(subarray)
                        # Call the save image function
                        im.save(out_npath)
                        # Export metadata as csv for downstream processing
                        # First, get the name of the original reference file to search on...
                        # Then, record this file's identity and metadata of interest
                        nd = {}
                        nd['ref_key'] = container_2D
                        nd['mask_id'] = filename
                        nd['mask_path'] = out_npath
                        nd['ob_class'] = otype
                        nd['ob_user'] = user
                        nd['centroidX'] = cx
                        nd['centroidY'] = cy
                        nd['object_area'] = area
                        nd['object_radius'] = radius
                        nd['x0'] = x0_val
                        nd['x1'] = x1_val
                        nd['y0'] = y0_val
                        nd['y1'] = y1_val
                        # Create output CSV for downstream aggregation
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path_to_new_file = 'project_MGNfelis_Oct2021_PatchesTraining.csv'
                        if os.path.exists(output_path_to_new_file):
                            new_df.to_csv(output_path_to_new_file, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path_to_new_file, header=True, index=False)
                    elif (cy >= 1992) and (cy <= 2192):
                        # No offset
                        cy0 = 1792
                        cy1 = 2192
                        """ OUTPUT """
                        # Create Nomenclature
                        # Format the z-value to have 100000s place (for downstream sorting on number)
                        x0_val = '{:0>6d}'.format(cx0)
                        y0_val = '{:0>6d}'.format(cy0)
                        x1_val = '{:0>6d}'.format(cx1)
                        y1_val = '{:0>6d}'.format(cy1)
                        ob_subarray = 'y' + str(y0_val) + '_' + 'x' + str(x0_val)
                        object_identity =  ob_subarray + '_' + omdata
                        filename = container_2D + '_' + object_identity + '.png'
                        # Establish the path for the image when output
                        out_npath = os.path.join(odir, filename)
                        # Select the array to be saved from data
                        subarray = p_mask[cy0:cy1, cx0:cx1]
                        # Convert the array to an image
                        im = Image.fromarray(subarray)
                        # Call the save image function
                        im.save(out_npath)
                        # Export metadata as csv for downstream processing
                        # First, get the name of the original reference file to search on...
                        # Then, record this file's identity and metadata of interest
                        nd = {}
                        nd['ref_key'] = container_2D
                        nd['mask_id'] = filename
                        nd['mask_path'] = out_npath
                        nd['ob_class'] = otype
                        nd['ob_user'] = user
                        nd['centroidX'] = cx
                        nd['centroidY'] = cy
                        nd['object_area'] = area
                        nd['object_radius'] = radius
                        nd['x0'] = x0_val
                        nd['x1'] = x1_val
                        nd['y0'] = y0_val
                        nd['y1'] = y1_val
                        # Create output CSV for downstream aggregation
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path_to_new_file = 'project_MGNfelis_Oct2021_PatchesTraining.csv'
                        if os.path.exists(output_path_to_new_file):
                            new_df.to_csv(output_path_to_new_file, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path_to_new_file, header=True, index=False)
                    else:
                        print('wtf!!!', container, oid)
                        continue
                elif (cx >= 790) and (cx <= 1376):
                    # Set up conditional for x values
                    # Offset (as positive for first half of image dim)
                    # Calculate the residual (i.e. distance from closest threshold) to dynamically set the offset
                    prev_bound = 789
                    res = cx - prev_bound
                    off_set = round(res*(0.25))
                    # Set the upper coordinate
                    cx0 = (cx - 200) + off_set
                    cx1 = (cx + 200) + off_set
                    if (cy >= 0) and (cy <= 200):
                        # No offset
                        cy0 = 0
                        cy1 = 400
                        """ OUTPUT """
                        # Create Nomenclature
                        # Format the z-value to have 100000s place (for downstream sorting on number)
                        x0_val = '{:0>6d}'.format(cx0)
                        y0_val = '{:0>6d}'.format(cy0)
                        x1_val = '{:0>6d}'.format(cx1)
                        y1_val = '{:0>6d}'.format(cy1)
                        ob_subarray = 'y' + str(y0_val) + '_' + 'x' + str(x0_val)
                        object_identity =  ob_subarray + '_' + omdata
                        filename = container_2D + '_' + object_identity + '.png'
                        # Establish the path for the image when output
                        out_npath = os.path.join(odir, filename)
                        # Select the array to be saved from data
                        subarray = p_mask[cy0:cy1, cx0:cx1]
                        # Convert the array to an image
                        im = Image.fromarray(subarray)
                        # Call the save image function
                        im.save(out_npath)
                        # Export metadata as csv for downstream processing
                        # First, get the name of the original reference file to search on...
                        # Then, record this file's identity and metadata of interest
                        nd = {}
                        nd['ref_key'] = container_2D
                        nd['mask_id'] = filename
                        nd['mask_path'] = out_npath
                        nd['ob_class'] = otype
                        nd['ob_user'] = user
                        nd['centroidX'] = cx
                        nd['centroidY'] = cy
                        nd['object_area'] = area
                        nd['object_radius'] = radius
                        nd['x0'] = x0_val
                        nd['x1'] = x1_val
                        nd['y0'] = y0_val
                        nd['y1'] = y1_val
                        # Create output CSV for downstream aggregation
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path_to_new_file = 'project_MGNfelis_Oct2021_PatchesTraining.csv'
                        if os.path.exists(output_path_to_new_file):
                            new_df.to_csv(output_path_to_new_file, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path_to_new_file, header=True, index=False)
                    elif (cy >= 201) and (cy <= 649):
                        # Offset (as positive for first half of image dim)
                        # Calculate the residual (i.e. distance from closest threshold) to dynamically set the offset
                        prev_bound = 200
                        res = cy - prev_bound
                        off_set = round(res*(0.25))
                        # Set the upper coordinate
                        cy0 = (cy - 200) + off_set
                        cy1 = (cy + 200) + off_set
                        """ OUTPUT """
                        # Create Nomenclature
                        # Format the z-value to have 100000s place (for downstream sorting on number)
                        x0_val = '{:0>6d}'.format(cx0)
                        y0_val = '{:0>6d}'.format(cy0)
                        x1_val = '{:0>6d}'.format(cx1)
                        y1_val = '{:0>6d}'.format(cy1)
                        ob_subarray = 'y' + str(y0_val) + '_' + 'x' + str(x0_val)
                        object_identity =  ob_subarray + '_' + omdata
                        filename = container_2D + '_' + object_identity + '.png'
                        # Establish the path for the image when output
                        out_npath = os.path.join(odir, filename)
                        # Select the array to be saved from data
                        subarray = p_mask[cy0:cy1, cx0:cx1]
                        # Convert the array to an image
                        im = Image.fromarray(subarray)
                        # Call the save image function
                        im.save(out_npath)
                        # Export metadata as csv for downstream processing
                        # First, get the name of the original reference file to search on...
                        # Then, record this file's identity and metadata of interest
                        nd = {}
                        nd['ref_key'] = container_2D
                        nd['mask_id'] = filename
                        nd['mask_path'] = out_npath
                        nd['ob_class'] = otype
                        nd['ob_user'] = user
                        nd['centroidX'] = cx
                        nd['centroidY'] = cy
                        nd['object_area'] = area
                        nd['object_radius'] = radius
                        nd['x0'] = x0_val
                        nd['x1'] = x1_val
                        nd['y0'] = y0_val
                        nd['y1'] = y1_val
                        # Create output CSV for downstream aggregation
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path_to_new_file = 'project_MGNfelis_Oct2021_PatchesTraining.csv'
                        if os.path.exists(output_path_to_new_file):
                            new_df.to_csv(output_path_to_new_file, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path_to_new_file, header=True, index=False)
                    elif (cy >= 650) and (cy <= 1099):
                        # Offset (as positive for first half of image dim)
                        # Calculate the residual (i.e. distance from closest threshold) to dynamically set the offset
                        prev_bound = 649
                        res = cy - prev_bound
                        off_set = round(res*(0.25))
                        # Set the upper coordinate
                        cy0 = (cy - 200) + off_set
                        cy1 = (cy + 200) + off_set
                        """ OUTPUT """
                        # Create Nomenclature
                        # Format the z-value to have 100000s place (for downstream sorting on number)
                        x0_val = '{:0>6d}'.format(cx0)
                        y0_val = '{:0>6d}'.format(cy0)
                        x1_val = '{:0>6d}'.format(cx1)
                        y1_val = '{:0>6d}'.format(cy1)
                        ob_subarray = 'y' + str(y0_val) + '_' + 'x' + str(x0_val)
                        object_identity =  ob_subarray + '_' + omdata
                        filename = container_2D + '_' + object_identity + '.png'
                        # Establish the path for the image when output
                        out_npath = os.path.join(odir, filename)
                        # Select the array to be saved from data
                        subarray = p_mask[cy0:cy1, cx0:cx1]
                        # Convert the array to an image
                        im = Image.fromarray(subarray)
                        # Call the save image function
                        im.save(out_npath)
                        # Export metadata as csv for downstream processing
                        # First, get the name of the original reference file to search on...
                        # Then, record this file's identity and metadata of interest
                        nd = {}
                        nd['ref_key'] = container_2D
                        nd['mask_id'] = filename
                        nd['mask_path'] = out_npath
                        nd['ob_class'] = otype
                        nd['ob_user'] = user
                        nd['centroidX'] = cx
                        nd['centroidY'] = cy
                        nd['object_area'] = area
                        nd['object_radius'] = radius
                        nd['x0'] = x0_val
                        nd['x1'] = x1_val
                        nd['y0'] = y0_val
                        nd['y1'] = y1_val
                        # Create output CSV for downstream aggregation
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path_to_new_file = 'project_MGNfelis_Oct2021_PatchesTraining.csv'
                        if os.path.exists(output_path_to_new_file):
                            new_df.to_csv(output_path_to_new_file, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path_to_new_file, header=True, index=False)
                    elif (cy >= 1100) and (cy <= 1548):
                        # Offset (as negative for second half of image dim)
                        # Calculate the residual (i.e. distance from closest threshold) to dynamically set the offset
                        prev_bound = 1099
                        res = cy - prev_bound
                        off_set = round(res*(0.25))
                        # Set the upper coordinate
                        cy0 = (cy - 200) - off_set
                        cy1 = (cy + 200) - off_set
                        """ OUTPUT """
                        # Create Nomenclature
                        # Format the z-value to have 100000s place (for downstream sorting on number)
                        x0_val = '{:0>6d}'.format(cx0)
                        y0_val = '{:0>6d}'.format(cy0)
                        x1_val = '{:0>6d}'.format(cx1)
                        y1_val = '{:0>6d}'.format(cy1)
                        ob_subarray = 'y' + str(y0_val) + '_' + 'x' + str(x0_val)
                        object_identity =  ob_subarray + '_' + omdata
                        filename = container_2D + '_' + object_identity + '.png'
                        # Establish the path for the image when output
                        out_npath = os.path.join(odir, filename)
                        # Select the array to be saved from data
                        subarray = p_mask[cy0:cy1, cx0:cx1]
                        # Convert the array to an image
                        im = Image.fromarray(subarray)
                        # Call the save image function
                        im.save(out_npath)
                        # Export metadata as csv for downstream processing
                        # First, get the name of the original reference file to search on...
                        # Then, record this file's identity and metadata of interest
                        nd = {}
                        nd['ref_key'] = container_2D
                        nd['mask_id'] = filename
                        nd['mask_path'] = out_npath
                        nd['ob_class'] = otype
                        nd['ob_user'] = user
                        nd['centroidX'] = cx
                        nd['centroidY'] = cy
                        nd['object_area'] = area
                        nd['object_radius'] = radius
                        nd['x0'] = x0_val
                        nd['x1'] = x1_val
                        nd['y0'] = y0_val
                        nd['y1'] = y1_val
                        # Create output CSV for downstream aggregation
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path_to_new_file = 'project_MGNfelis_Oct2021_PatchesTraining.csv'
                        if os.path.exists(output_path_to_new_file):
                            new_df.to_csv(output_path_to_new_file, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path_to_new_file, header=True, index=False)
                    elif (cy >= 1549) and (cy <= 1991):
                        # Offset (as negative for second half of image dim)
                        # Calculate the residual (i.e. distance from closest threshold) to dynamically set the offset
                        prev_bound = 1548
                        res = cy - prev_bound
                        off_set = round(res*(0.25))
                        # Set the upper coordinate
                        cy0 = (cy - 200) - off_set
                        cy1 = (cy + 200) - off_set
                        """ OUTPUT """
                        # Create Nomenclature
                        # Format the z-value to have 100000s place (for downstream sorting on number)
                        x0_val = '{:0>6d}'.format(cx0)
                        y0_val = '{:0>6d}'.format(cy0)
                        x1_val = '{:0>6d}'.format(cx1)
                        y1_val = '{:0>6d}'.format(cy1)
                        ob_subarray = 'y' + str(y0_val) + '_' + 'x' + str(x0_val)
                        object_identity =  ob_subarray + '_' + omdata
                        filename = container_2D + '_' + object_identity + '.png'
                        # Establish the path for the image when output
                        out_npath = os.path.join(odir, filename)
                        # Select the array to be saved from data
                        subarray = p_mask[cy0:cy1, cx0:cx1]
                        # Convert the array to an image
                        im = Image.fromarray(subarray)
                        # Call the save image function
                        im.save(out_npath)
                        # Export metadata as csv for downstream processing
                        # First, get the name of the original reference file to search on...
                        # Then, record this file's identity and metadata of interest
                        nd = {}
                        nd['ref_key'] = container_2D
                        nd['mask_id'] = filename
                        nd['mask_path'] = out_npath
                        nd['ob_class'] = otype
                        nd['ob_user'] = user
                        nd['centroidX'] = cx
                        nd['centroidY'] = cy
                        nd['object_area'] = area
                        nd['object_radius'] = radius
                        nd['x0'] = x0_val
                        nd['x1'] = x1_val
                        nd['y0'] = y0_val
                        nd['y1'] = y1_val
                        # Create output CSV for downstream aggregation
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path_to_new_file = 'project_MGNfelis_Oct2021_PatchesTraining.csv'
                        if os.path.exists(output_path_to_new_file):
                            new_df.to_csv(output_path_to_new_file, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path_to_new_file, header=True, index=False)
                    elif (cy >= 1992) and (cy <= 2192):
                        # No offset
                        cy0 = 1792
                        cy1 = 2192
                        """ OUTPUT """
                        # Create Nomenclature
                        # Format the z-value to have 100000s place (for downstream sorting on number)
                        x0_val = '{:0>6d}'.format(cx0)
                        y0_val = '{:0>6d}'.format(cy0)
                        x1_val = '{:0>6d}'.format(cx1)
                        y1_val = '{:0>6d}'.format(cy1)
                        ob_subarray = 'y' + str(y0_val) + '_' + 'x' + str(x0_val)
                        object_identity =  ob_subarray + '_' + omdata
                        filename = container_2D + '_' + object_identity + '.png'
                        # Establish the path for the image when output
                        out_npath = os.path.join(odir, filename)
                        # Select the array to be saved from data
                        subarray = p_mask[cy0:cy1, cx0:cx1]
                        # Convert the array to an image
                        im = Image.fromarray(subarray)
                        # Call the save image function
                        im.save(out_npath)
                        # Export metadata as csv for downstream processing
                        # First, get the name of the original reference file to search on...
                        # Then, record this file's identity and metadata of interest
                        nd = {}
                        nd['ref_key'] = container_2D
                        nd['mask_id'] = filename
                        nd['mask_path'] = out_npath
                        nd['ob_class'] = otype
                        nd['ob_user'] = user
                        nd['centroidX'] = cx
                        nd['centroidY'] = cy
                        nd['object_area'] = area
                        nd['object_radius'] = radius
                        nd['x0'] = x0_val
                        nd['x1'] = x1_val
                        nd['y0'] = y0_val
                        nd['y1'] = y1_val
                        # Create output CSV for downstream aggregation
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path_to_new_file = 'project_MGNfelis_Oct2021_PatchesTraining.csv'
                        if os.path.exists(output_path_to_new_file):
                            new_df.to_csv(output_path_to_new_file, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path_to_new_file, header=True, index=False)
                    else:
                        print('wtf!!!', container, oid)
                        continue
                elif (cx >= 1377) and (cx <= 1964):
                    # Set up conditional for x values
                    # Offset (as negative for second half of image dim)
                    # Calculate the residual (i.e. distance from closest threshold) to dynamically set the offset
                    prev_bound = 1376
                    res = cx - prev_bound
                    off_set = round(res*(0.25))
                    # Set the upper coordinate
                    cx0 = (cx - 200) - off_set
                    cx1 = (cx + 200) - off_set
                    if (cy >= 0) and (cy <= 200):
                        # No offset
                        cy0 = 0
                        cy1 = 400
                        """ OUTPUT """
                        # Create Nomenclature
                        # Format the z-value to have 100000s place (for downstream sorting on number)
                        x0_val = '{:0>6d}'.format(cx0)
                        y0_val = '{:0>6d}'.format(cy0)
                        x1_val = '{:0>6d}'.format(cx1)
                        y1_val = '{:0>6d}'.format(cy1)
                        ob_subarray = 'y' + str(y0_val) + '_' + 'x' + str(x0_val)
                        object_identity =  ob_subarray + '_' + omdata
                        filename = container_2D + '_' + object_identity + '.png'
                        # Establish the path for the image when output
                        out_npath = os.path.join(odir, filename)
                        # Select the array to be saved from data
                        subarray = p_mask[cy0:cy1, cx0:cx1]
                        # Convert the array to an image
                        im = Image.fromarray(subarray)
                        # Call the save image function
                        im.save(out_npath)
                        # Export metadata as csv for downstream processing
                        # First, get the name of the original reference file to search on...
                        # Then, record this file's identity and metadata of interest
                        nd = {}
                        nd['ref_key'] = container_2D
                        nd['mask_id'] = filename
                        nd['mask_path'] = out_npath
                        nd['ob_class'] = otype
                        nd['ob_user'] = user
                        nd['centroidX'] = cx
                        nd['centroidY'] = cy
                        nd['object_area'] = area
                        nd['object_radius'] = radius
                        nd['x0'] = x0_val
                        nd['x1'] = x1_val
                        nd['y0'] = y0_val
                        nd['y1'] = y1_val
                        # Create output CSV for downstream aggregation
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path_to_new_file = 'project_MGNfelis_Oct2021_PatchesTraining.csv'
                        if os.path.exists(output_path_to_new_file):
                            new_df.to_csv(output_path_to_new_file, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path_to_new_file, header=True, index=False)
                    elif (cy >= 201) and (cy <= 649):
                        # Offset (as positive for first half of image dim)
                        # Calculate the residual (i.e. distance from closest threshold) to dynamically set the offset
                        prev_bound = 200
                        res = cy - prev_bound
                        off_set = round(res*(0.25))
                        # Set the upper coordinate
                        cy0 = (cy - 200) + off_set
                        cy1 = (cy + 200) + off_set
                        """ OUTPUT """
                        # Create Nomenclature
                        # Format the z-value to have 100000s place (for downstream sorting on number)
                        x0_val = '{:0>6d}'.format(cx0)
                        y0_val = '{:0>6d}'.format(cy0)
                        x1_val = '{:0>6d}'.format(cx1)
                        y1_val = '{:0>6d}'.format(cy1)
                        ob_subarray = 'y' + str(y0_val) + '_' + 'x' + str(x0_val)
                        object_identity =  ob_subarray + '_' + omdata
                        filename = container_2D + '_' + object_identity + '.png'
                        # Establish the path for the image when output
                        out_npath = os.path.join(odir, filename)
                        # Select the array to be saved from data
                        subarray = p_mask[cy0:cy1, cx0:cx1]
                        # Convert the array to an image
                        im = Image.fromarray(subarray)
                        # Call the save image function
                        im.save(out_npath)
                        # Export metadata as csv for downstream processing
                        # First, get the name of the original reference file to search on...
                        # Then, record this file's identity and metadata of interest
                        nd = {}
                        nd['ref_key'] = container_2D
                        nd['mask_id'] = filename
                        nd['mask_path'] = out_npath
                        nd['ob_class'] = otype
                        nd['ob_user'] = user
                        nd['centroidX'] = cx
                        nd['centroidY'] = cy
                        nd['object_area'] = area
                        nd['object_radius'] = radius
                        nd['x0'] = x0_val
                        nd['x1'] = x1_val
                        nd['y0'] = y0_val
                        nd['y1'] = y1_val
                        # Create output CSV for downstream aggregation
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path_to_new_file = 'project_MGNfelis_Oct2021_PatchesTraining.csv'
                        if os.path.exists(output_path_to_new_file):
                            new_df.to_csv(output_path_to_new_file, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path_to_new_file, header=True, index=False)
                    elif (cy >= 650) and (cy <= 1099):
                        # Offset (as positive for first half of image dim)
                        # Calculate the residual (i.e. distance from closest threshold) to dynamically set the offset
                        prev_bound = 649
                        res = cy - prev_bound
                        off_set = round(res*(0.25))
                        # Set the upper coordinate
                        cy0 = (cy - 200) + off_set
                        cy1 = (cy + 200) + off_set
                        """ OUTPUT """
                        # Create Nomenclature
                        # Format the z-value to have 100000s place (for downstream sorting on number)
                        x0_val = '{:0>6d}'.format(cx0)
                        y0_val = '{:0>6d}'.format(cy0)
                        x1_val = '{:0>6d}'.format(cx1)
                        y1_val = '{:0>6d}'.format(cy1)
                        ob_subarray = 'y' + str(y0_val) + '_' + 'x' + str(x0_val)
                        object_identity =  ob_subarray + '_' + omdata
                        filename = container_2D + '_' + object_identity + '.png'
                        # Establish the path for the image when output
                        out_npath = os.path.join(odir, filename)
                        # Select the array to be saved from data
                        subarray = p_mask[cy0:cy1, cx0:cx1]
                        # Convert the array to an image
                        im = Image.fromarray(subarray)
                        # Call the save image function
                        im.save(out_npath)
                        # Export metadata as csv for downstream processing
                        # First, get the name of the original reference file to search on...
                        # Then, record this file's identity and metadata of interest
                        nd = {}
                        nd['ref_key'] = container_2D
                        nd['mask_id'] = filename
                        nd['mask_path'] = out_npath
                        nd['ob_class'] = otype
                        nd['ob_user'] = user
                        nd['centroidX'] = cx
                        nd['centroidY'] = cy
                        nd['object_area'] = area
                        nd['object_radius'] = radius
                        nd['x0'] = x0_val
                        nd['x1'] = x1_val
                        nd['y0'] = y0_val
                        nd['y1'] = y1_val
                        # Create output CSV for downstream aggregation
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path_to_new_file = 'project_MGNfelis_Oct2021_PatchesTraining.csv'
                        if os.path.exists(output_path_to_new_file):
                            new_df.to_csv(output_path_to_new_file, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path_to_new_file, header=True, index=False)
                    elif (cy >= 1100) and (cy <= 1548):
                        # Offset (as negative for second half of image dim)
                        # Calculate the residual (i.e. distance from closest threshold) to dynamically set the offset
                        prev_bound = 1099
                        res = cy - prev_bound
                        off_set = round(res*(0.25))
                        # Set the upper coordinate
                        cy0 = (cy - 200) - off_set
                        cy1 = (cy + 200) - off_set
                        """ OUTPUT """
                        # Create Nomenclature
                        # Format the z-value to have 100000s place (for downstream sorting on number)
                        x0_val = '{:0>6d}'.format(cx0)
                        y0_val = '{:0>6d}'.format(cy0)
                        x1_val = '{:0>6d}'.format(cx1)
                        y1_val = '{:0>6d}'.format(cy1)
                        ob_subarray = 'y' + str(y0_val) + '_' + 'x' + str(x0_val)
                        object_identity =  ob_subarray + '_' + omdata
                        filename = container_2D + '_' + object_identity + '.png'
                        # Establish the path for the image when output
                        out_npath = os.path.join(odir, filename)
                        # Select the array to be saved from data
                        subarray = p_mask[cy0:cy1, cx0:cx1]
                        # Convert the array to an image
                        im = Image.fromarray(subarray)
                        # Call the save image function
                        im.save(out_npath)
                        # Export metadata as csv for downstream processing
                        # First, get the name of the original reference file to search on...
                        # Then, record this file's identity and metadata of interest
                        nd = {}
                        nd['ref_key'] = container_2D
                        nd['mask_id'] = filename
                        nd['mask_path'] = out_npath
                        nd['ob_class'] = otype
                        nd['ob_user'] = user
                        nd['centroidX'] = cx
                        nd['centroidY'] = cy
                        nd['object_area'] = area
                        nd['object_radius'] = radius
                        nd['x0'] = x0_val
                        nd['x1'] = x1_val
                        nd['y0'] = y0_val
                        nd['y1'] = y1_val
                        # Create output CSV for downstream aggregation
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path_to_new_file = 'project_MGNfelis_Oct2021_PatchesTraining.csv'
                        if os.path.exists(output_path_to_new_file):
                            new_df.to_csv(output_path_to_new_file, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path_to_new_file, header=True, index=False)
                    elif (cy >= 1549) and (cy <= 1991):
                        # Offset (as negative for second half of image dim)
                        # Calculate the residual (i.e. distance from closest threshold) to dynamically set the offset
                        prev_bound = 1548
                        res = cy - prev_bound
                        off_set = round(res*(0.25))
                        # Set the upper coordinate
                        cy0 = (cy - 200) - off_set
                        cy1 = (cy + 200) - off_set
                        """ OUTPUT """
                        # Create Nomenclature
                        # Format the z-value to have 100000s place (for downstream sorting on number)
                        x0_val = '{:0>6d}'.format(cx0)
                        y0_val = '{:0>6d}'.format(cy0)
                        x1_val = '{:0>6d}'.format(cx1)
                        y1_val = '{:0>6d}'.format(cy1)
                        ob_subarray = 'y' + str(y0_val) + '_' + 'x' + str(x0_val)
                        object_identity =  ob_subarray + '_' + omdata
                        filename = container_2D + '_' + object_identity + '.png'
                        # Establish the path for the image when output
                        out_npath = os.path.join(odir, filename)
                        # Select the array to be saved from data
                        subarray = p_mask[cy0:cy1, cx0:cx1]
                        # Convert the array to an image
                        im = Image.fromarray(subarray)
                        # Call the save image function
                        im.save(out_npath)
                        # Export metadata as csv for downstream processing
                        # First, get the name of the original reference file to search on...
                        # Then, record this file's identity and metadata of interest
                        nd = {}
                        nd['ref_key'] = container_2D
                        nd['mask_id'] = filename
                        nd['mask_path'] = out_npath
                        nd['ob_class'] = otype
                        nd['ob_user'] = user
                        nd['centroidX'] = cx
                        nd['centroidY'] = cy
                        nd['object_area'] = area
                        nd['object_radius'] = radius
                        nd['x0'] = x0_val
                        nd['x1'] = x1_val
                        nd['y0'] = y0_val
                        nd['y1'] = y1_val
                        # Create output CSV for downstream aggregation
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path_to_new_file = 'project_MGNfelis_Oct2021_PatchesTraining.csv'
                        if os.path.exists(output_path_to_new_file):
                            new_df.to_csv(output_path_to_new_file, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path_to_new_file, header=True, index=False)
                    elif (cy >= 1992) and (cy <= 2192):
                        # No offset
                        cy0 = 1792
                        cy1 = 2192
                        """ OUTPUT """
                        # Create Nomenclature
                        # Format the z-value to have 100000s place (for downstream sorting on number)
                        x0_val = '{:0>6d}'.format(cx0)
                        y0_val = '{:0>6d}'.format(cy0)
                        x1_val = '{:0>6d}'.format(cx1)
                        y1_val = '{:0>6d}'.format(cy1)
                        ob_subarray = 'y' + str(y0_val) + '_' + 'x' + str(x0_val)
                        object_identity =  ob_subarray + '_' + omdata
                        filename = container_2D + '_' + object_identity + '.png'
                        # Establish the path for the image when output
                        out_npath = os.path.join(odir, filename)
                        # Select the array to be saved from data
                        subarray = p_mask[cy0:cy1, cx0:cx1]
                        # Convert the array to an image
                        im = Image.fromarray(subarray)
                        # Call the save image function
                        im.save(out_npath)
                        # Export metadata as csv for downstream processing
                        # First, get the name of the original reference file to search on...
                        # Then, record this file's identity and metadata of interest
                        nd = {}
                        nd['ref_key'] = container_2D
                        nd['mask_id'] = filename
                        nd['mask_path'] = out_npath
                        nd['ob_class'] = otype
                        nd['ob_user'] = user
                        nd['centroidX'] = cx
                        nd['centroidY'] = cy
                        nd['object_area'] = area
                        nd['object_radius'] = radius
                        nd['x0'] = x0_val
                        nd['x1'] = x1_val
                        nd['y0'] = y0_val
                        nd['y1'] = y1_val
                        # Create output CSV for downstream aggregation
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path_to_new_file = 'project_MGNfelis_Oct2021_PatchesTraining.csv'
                        if os.path.exists(output_path_to_new_file):
                            new_df.to_csv(output_path_to_new_file, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path_to_new_file, header=True, index=False)
                    else:
                        print('wtf!!!', container, oid)
                        continue
                elif (cx >= 1965) and (cx <= 2551):
                    # Set up conditional for x values
                    # Offset (as negative for second half of image dim)
                    # Calculate the residual (i.e. distance from closest threshold) to dynamically set the offset
                    prev_bound = 1964
                    res = cx - prev_bound
                    off_set = round(res*(0.25))
                    # Set the upper coordinate
                    cx0 = (cx - 200) - off_set
                    cx1 = (cx + 200) - off_set
                    if (cy >= 0) and (cy <= 200):
                        # No offset
                        cy0 = 0
                        cy1 = 400
                        """ OUTPUT """
                        # Create Nomenclature
                        # Format the z-value to have 100000s place (for downstream sorting on number)
                        x0_val = '{:0>6d}'.format(cx0)
                        y0_val = '{:0>6d}'.format(cy0)
                        x1_val = '{:0>6d}'.format(cx1)
                        y1_val = '{:0>6d}'.format(cy1)
                        ob_subarray = 'y' + str(y0_val) + '_' + 'x' + str(x0_val)
                        object_identity =  ob_subarray + '_' + omdata
                        filename = container_2D + '_' + object_identity + '.png'
                        # Establish the path for the image when output
                        out_npath = os.path.join(odir, filename)
                        # Select the array to be saved from data
                        subarray = p_mask[cy0:cy1, cx0:cx1]
                        # Convert the array to an image
                        im = Image.fromarray(subarray)
                        # Call the save image function
                        im.save(out_npath)
                        # Export metadata as csv for downstream processing
                        # First, get the name of the original reference file to search on...
                        # Then, record this file's identity and metadata of interest
                        nd = {}
                        nd['ref_key'] = container_2D
                        nd['mask_id'] = filename
                        nd['mask_path'] = out_npath
                        nd['ob_class'] = otype
                        nd['ob_user'] = user
                        nd['centroidX'] = cx
                        nd['centroidY'] = cy
                        nd['object_area'] = area
                        nd['object_radius'] = radius
                        nd['x0'] = x0_val
                        nd['x1'] = x1_val
                        nd['y0'] = y0_val
                        nd['y1'] = y1_val
                        # Create output CSV for downstream aggregation
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path_to_new_file = 'project_MGNfelis_Oct2021_PatchesTraining.csv'
                        if os.path.exists(output_path_to_new_file):
                            new_df.to_csv(output_path_to_new_file, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path_to_new_file, header=True, index=False)
                    elif (cy >= 201) and (cy <= 649):
                        # Offset (as positive for first half of image dim)
                        # Calculate the residual (i.e. distance from closest threshold) to dynamically set the offset
                        prev_bound = 200
                        res = cy - prev_bound
                        off_set = round(res*(0.25))
                        # Set the upper coordinate
                        cy0 = (cy - 200) + off_set
                        cy1 = (cy + 200) + off_set
                        """ OUTPUT """
                        # Create Nomenclature
                        # Format the z-value to have 100000s place (for downstream sorting on number)
                        x0_val = '{:0>6d}'.format(cx0)
                        y0_val = '{:0>6d}'.format(cy0)
                        x1_val = '{:0>6d}'.format(cx1)
                        y1_val = '{:0>6d}'.format(cy1)
                        ob_subarray = 'y' + str(y0_val) + '_' + 'x' + str(x0_val)
                        object_identity =  ob_subarray + '_' + omdata
                        filename = container_2D + '_' + object_identity + '.png'
                        # Establish the path for the image when output
                        out_npath = os.path.join(odir, filename)
                        # Select the array to be saved from data
                        subarray = p_mask[cy0:cy1, cx0:cx1]
                        # Convert the array to an image
                        im = Image.fromarray(subarray)
                        # Call the save image function
                        im.save(out_npath)
                        # Export metadata as csv for downstream processing
                        # First, get the name of the original reference file to search on...
                        # Then, record this file's identity and metadata of interest
                        nd = {}
                        nd['ref_key'] = container_2D
                        nd['mask_id'] = filename
                        nd['mask_path'] = out_npath
                        nd['ob_class'] = otype
                        nd['ob_user'] = user
                        nd['centroidX'] = cx
                        nd['centroidY'] = cy
                        nd['object_area'] = area
                        nd['object_radius'] = radius
                        nd['x0'] = x0_val
                        nd['x1'] = x1_val
                        nd['y0'] = y0_val
                        nd['y1'] = y1_val
                        # Create output CSV for downstream aggregation
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path_to_new_file = 'project_MGNfelis_Oct2021_PatchesTraining.csv'
                        if os.path.exists(output_path_to_new_file):
                            new_df.to_csv(output_path_to_new_file, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path_to_new_file, header=True, index=False)
                    elif (cy >= 650) and (cy <= 1099):
                        # Offset (as positive for first half of image dim)
                        # Calculate the residual (i.e. distance from closest threshold) to dynamically set the offset
                        prev_bound = 649
                        res = cy - prev_bound
                        off_set = round(res*(0.25))
                        # Set the upper coordinate
                        cy0 = (cy - 200) + off_set
                        cy1 = (cy + 200) + off_set
                        """ OUTPUT """
                        # Create Nomenclature
                        # Format the z-value to have 100000s place (for downstream sorting on number)
                        x0_val = '{:0>6d}'.format(cx0)
                        y0_val = '{:0>6d}'.format(cy0)
                        x1_val = '{:0>6d}'.format(cx1)
                        y1_val = '{:0>6d}'.format(cy1)
                        ob_subarray = 'y' + str(y0_val) + '_' + 'x' + str(x0_val)
                        object_identity =  ob_subarray + '_' + omdata
                        filename = container_2D + '_' + object_identity + '.png'
                        # Establish the path for the image when output
                        out_npath = os.path.join(odir, filename)
                        # Select the array to be saved from data
                        subarray = p_mask[cy0:cy1, cx0:cx1]
                        # Convert the array to an image
                        im = Image.fromarray(subarray)
                        # Call the save image function
                        im.save(out_npath)
                        # Export metadata as csv for downstream processing
                        # First, get the name of the original reference file to search on...
                        # Then, record this file's identity and metadata of interest
                        nd = {}
                        nd['ref_key'] = container_2D
                        nd['mask_id'] = filename
                        nd['mask_path'] = out_npath
                        nd['ob_class'] = otype
                        nd['ob_user'] = user
                        nd['centroidX'] = cx
                        nd['centroidY'] = cy
                        nd['object_area'] = area
                        nd['object_radius'] = radius
                        nd['x0'] = x0_val
                        nd['x1'] = x1_val
                        nd['y0'] = y0_val
                        nd['y1'] = y1_val
                        # Create output CSV for downstream aggregation
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path_to_new_file = 'project_MGNfelis_Oct2021_PatchesTraining.csv'
                        if os.path.exists(output_path_to_new_file):
                            new_df.to_csv(output_path_to_new_file, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path_to_new_file, header=True, index=False)
                    elif (cy >= 1100) and (cy <= 1548):
                        # Offset (as negative for second half of image dim)
                        # Calculate the residual (i.e. distance from closest threshold) to dynamically set the offset
                        prev_bound = 1099
                        res = cy - prev_bound
                        off_set = round(res*(0.25))
                        # Set the upper coordinate
                        cy0 = (cy - 200) - off_set
                        cy1 = (cy + 200) - off_set
                        """ OUTPUT """
                        # Create Nomenclature
                        # Format the z-value to have 100000s place (for downstream sorting on number)
                        x0_val = '{:0>6d}'.format(cx0)
                        y0_val = '{:0>6d}'.format(cy0)
                        x1_val = '{:0>6d}'.format(cx1)
                        y1_val = '{:0>6d}'.format(cy1)
                        ob_subarray = 'y' + str(y0_val) + '_' + 'x' + str(x0_val)
                        object_identity =  ob_subarray + '_' + omdata
                        filename = container_2D + '_' + object_identity + '.png'
                        # Establish the path for the image when output
                        out_npath = os.path.join(odir, filename)
                        # Select the array to be saved from data
                        subarray = p_mask[cy0:cy1, cx0:cx1]
                        # Convert the array to an image
                        im = Image.fromarray(subarray)
                        # Call the save image function
                        im.save(out_npath)
                        # Export metadata as csv for downstream processing
                        # First, get the name of the original reference file to search on...
                        # Then, record this file's identity and metadata of interest
                        nd = {}
                        nd['ref_key'] = container_2D
                        nd['mask_id'] = filename
                        nd['mask_path'] = out_npath
                        nd['ob_class'] = otype
                        nd['ob_user'] = user
                        nd['centroidX'] = cx
                        nd['centroidY'] = cy
                        nd['object_area'] = area
                        nd['object_radius'] = radius
                        nd['x0'] = x0_val
                        nd['x1'] = x1_val
                        nd['y0'] = y0_val
                        nd['y1'] = y1_val
                        # Create output CSV for downstream aggregation
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path_to_new_file = 'project_MGNfelis_Oct2021_PatchesTraining.csv'
                        if os.path.exists(output_path_to_new_file):
                            new_df.to_csv(output_path_to_new_file, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path_to_new_file, header=True, index=False)
                    elif (cy >= 1549) and (cy <= 1991):
                        # Offset (as negative for second half of image dim)
                        # Calculate the residual (i.e. distance from closest threshold) to dynamically set the offset
                        prev_bound = 1548
                        res = cy - prev_bound
                        off_set = round(res*(0.25))
                        # Set the upper coordinate
                        cy0 = (cy - 200) - off_set
                        cy1 = (cy + 200) - off_set
                        """ OUTPUT """
                        # Create Nomenclature
                        # Format the z-value to have 100000s place (for downstream sorting on number)
                        x0_val = '{:0>6d}'.format(cx0)
                        y0_val = '{:0>6d}'.format(cy0)
                        x1_val = '{:0>6d}'.format(cx1)
                        y1_val = '{:0>6d}'.format(cy1)
                        ob_subarray = 'y' + str(y0_val) + '_' + 'x' + str(x0_val)
                        object_identity =  ob_subarray + '_' + omdata
                        filename = container_2D + '_' + object_identity + '.png'
                        # Establish the path for the image when output
                        out_npath = os.path.join(odir, filename)
                        # Select the array to be saved from data
                        subarray = p_mask[cy0:cy1, cx0:cx1]
                        # Convert the array to an image
                        im = Image.fromarray(subarray)
                        # Call the save image function
                        im.save(out_npath)
                        # Export metadata as csv for downstream processing
                        # First, get the name of the original reference file to search on...
                        # Then, record this file's identity and metadata of interest
                        nd = {}
                        nd['ref_key'] = container_2D
                        nd['mask_id'] = filename
                        nd['mask_path'] = out_npath
                        nd['ob_class'] = otype
                        nd['ob_user'] = user
                        nd['centroidX'] = cx
                        nd['centroidY'] = cy
                        nd['object_area'] = area
                        nd['object_radius'] = radius
                        nd['x0'] = x0_val
                        nd['x1'] = x1_val
                        nd['y0'] = y0_val
                        nd['y1'] = y1_val
                        # Create output CSV for downstream aggregation
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path_to_new_file = 'project_MGNfelis_Oct2021_PatchesTraining.csv'
                        if os.path.exists(output_path_to_new_file):
                            new_df.to_csv(output_path_to_new_file, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path_to_new_file, header=True, index=False)
                    elif (cy >= 1992) and (cy <= 2192):
                        # No offset
                        cy0 = 1792
                        cy1 = 2192
                        """ OUTPUT """
                        # Create Nomenclature
                        # Format the z-value to have 100000s place (for downstream sorting on number)
                        x0_val = '{:0>6d}'.format(cx0)
                        y0_val = '{:0>6d}'.format(cy0)
                        x1_val = '{:0>6d}'.format(cx1)
                        y1_val = '{:0>6d}'.format(cy1)
                        ob_subarray = 'y' + str(y0_val) + '_' + 'x' + str(x0_val)
                        object_identity =  ob_subarray + '_' + omdata
                        filename = container_2D + '_' + object_identity + '.png'
                        # Establish the path for the image when output
                        out_npath = os.path.join(odir, filename)
                        # Select the array to be saved from data
                        subarray = p_mask[cy0:cy1, cx0:cx1]
                        # Convert the array to an image
                        im = Image.fromarray(subarray)
                        # Call the save image function
                        im.save(out_npath)
                        # Export metadata as csv for downstream processing
                        # First, get the name of the original reference file to search on...
                        # Then, record this file's identity and metadata of interest
                        nd = {}
                        nd['ref_key'] = container_2D
                        nd['mask_id'] = filename
                        nd['mask_path'] = out_npath
                        nd['ob_class'] = otype
                        nd['ob_user'] = user
                        nd['centroidX'] = cx
                        nd['centroidY'] = cy
                        nd['object_area'] = area
                        nd['object_radius'] = radius
                        nd['x0'] = x0_val
                        nd['x1'] = x1_val
                        nd['y0'] = y0_val
                        nd['y1'] = y1_val
                        # Create output CSV for downstream aggregation
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path_to_new_file = 'project_MGNfelis_Oct2021_PatchesTraining.csv'
                        if os.path.exists(output_path_to_new_file):
                            new_df.to_csv(output_path_to_new_file, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path_to_new_file, header=True, index=False)
                    else:
                        print('wtf!!!', container, oid)
                        continue
                elif (cx >= 2552) and (cx <= 2752):
                    # Edge case centroid coordinates are taken 'as-is' - no offset for X values
                    # So just go straight to building the options for cy
                    # For edge case X values... no need to offset since they already are! :) ...
                    cx0 = 2352
                    cx1 = 2752
                    if (cy >= 0) and (cy <= 200):
                        # No offset
                        cy0 = 0
                        cy1 = 400
                        """ OUTPUT """
                        # Create Nomenclature
                        # Format the z-value to have 100000s place (for downstream sorting on number)
                        x0_val = '{:0>6d}'.format(cx0)
                        y0_val = '{:0>6d}'.format(cy0)
                        x1_val = '{:0>6d}'.format(cx1)
                        y1_val = '{:0>6d}'.format(cy1)
                        ob_subarray = 'y' + str(y0_val) + '_' + 'x' + str(x0_val)
                        object_identity =  ob_subarray + '_' + omdata
                        filename = container_2D + '_' + object_identity + '.png'
                        # Establish the path for the image when output
                        out_npath = os.path.join(odir, filename)
                        # Select the array to be saved from data
                        subarray = p_mask[cy0:cy1, cx0:cx1]
                        # Convert the array to an image
                        im = Image.fromarray(subarray)
                        # Call the save image function
                        im.save(out_npath)
                        # Export metadata as csv for downstream processing
                        # First, get the name of the original reference file to search on...
                        # Then, record this file's identity and metadata of interest
                        nd = {}
                        nd['ref_key'] = container_2D
                        nd['mask_id'] = filename
                        nd['mask_path'] = out_npath
                        nd['ob_class'] = otype
                        nd['ob_user'] = user
                        nd['centroidX'] = cx
                        nd['centroidY'] = cy
                        nd['object_area'] = area
                        nd['object_radius'] = radius
                        nd['x0'] = x0_val
                        nd['x1'] = x1_val
                        nd['y0'] = y0_val
                        nd['y1'] = y1_val
                        # Create output CSV for downstream aggregation
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path_to_new_file = 'project_MGNfelis_Oct2021_PatchesTraining.csv'
                        if os.path.exists(output_path_to_new_file):
                            new_df.to_csv(output_path_to_new_file, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path_to_new_file, header=True, index=False)
                    elif (cy >= 201) and (cy <= 649):
                        # Offset (as positive for first half of image dim)
                        # Calculate the residual (i.e. distance from closest threshold) to dynamically set the offset
                        prev_bound = 200
                        res = cy - prev_bound
                        off_set = round(res*(0.25))
                        # Set the upper coordinate
                        cy0 = (cy - 200) + off_set
                        cy1 = (cy + 200) + off_set
                        """ OUTPUT """
                        # Create Nomenclature
                        # Format the z-value to have 100000s place (for downstream sorting on number)
                        x0_val = '{:0>6d}'.format(cx0)
                        y0_val = '{:0>6d}'.format(cy0)
                        x1_val = '{:0>6d}'.format(cx1)
                        y1_val = '{:0>6d}'.format(cy1)
                        ob_subarray = 'y' + str(y0_val) + '_' + 'x' + str(x0_val)
                        object_identity =  ob_subarray + '_' + omdata
                        filename = container_2D + '_' + object_identity + '.png'
                        # Establish the path for the image when output
                        out_npath = os.path.join(odir, filename)
                        # Select the array to be saved from data
                        subarray = p_mask[cy0:cy1, cx0:cx1]
                        # Convert the array to an image
                        im = Image.fromarray(subarray)
                        # Call the save image function
                        im.save(out_npath)
                        # Export metadata as csv for downstream processing
                        # First, get the name of the original reference file to search on...
                        # Then, record this file's identity and metadata of interest
                        nd = {}
                        nd['ref_key'] = container_2D
                        nd['mask_id'] = filename
                        nd['mask_path'] = out_npath
                        nd['ob_class'] = otype
                        nd['ob_user'] = user
                        nd['centroidX'] = cx
                        nd['centroidY'] = cy
                        nd['object_area'] = area
                        nd['object_radius'] = radius
                        nd['x0'] = x0_val
                        nd['x1'] = x1_val
                        nd['y0'] = y0_val
                        nd['y1'] = y1_val
                        # Create output CSV for downstream aggregation
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path_to_new_file = 'project_MGNfelis_Oct2021_PatchesTraining.csv'
                        if os.path.exists(output_path_to_new_file):
                            new_df.to_csv(output_path_to_new_file, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path_to_new_file, header=True, index=False)
                    elif (cy >= 650) and (cy <= 1099):
                        # Offset (as positive for first half of image dim)
                        # Calculate the residual (i.e. distance from closest threshold) to dynamically set the offset
                        prev_bound = 649
                        res = cy - prev_bound
                        off_set = round(res*(0.25))
                        # Set the upper coordinate
                        cy0 = (cy - 200) + off_set
                        cy1 = (cy + 200) + off_set
                        """ OUTPUT """
                        # Create Nomenclature
                        # Format the z-value to have 100000s place (for downstream sorting on number)
                        x0_val = '{:0>6d}'.format(cx0)
                        y0_val = '{:0>6d}'.format(cy0)
                        x1_val = '{:0>6d}'.format(cx1)
                        y1_val = '{:0>6d}'.format(cy1)
                        ob_subarray = 'y' + str(y0_val) + '_' + 'x' + str(x0_val)
                        object_identity =  ob_subarray + '_' + omdata
                        filename = container_2D + '_' + object_identity + '.png'
                        # Establish the path for the image when output
                        out_npath = os.path.join(odir, filename)
                        # Select the array to be saved from data
                        subarray = p_mask[cy0:cy1, cx0:cx1]
                        # Convert the array to an image
                        im = Image.fromarray(subarray)
                        # Call the save image function
                        im.save(out_npath)
                        # Export metadata as csv for downstream processing
                        # First, get the name of the original reference file to search on...
                        # Then, record this file's identity and metadata of interest
                        nd = {}
                        nd['ref_key'] = container_2D
                        nd['mask_id'] = filename
                        nd['mask_path'] = out_npath
                        nd['ob_class'] = otype
                        nd['ob_user'] = user
                        nd['centroidX'] = cx
                        nd['centroidY'] = cy
                        nd['object_area'] = area
                        nd['object_radius'] = radius
                        nd['x0'] = x0_val
                        nd['x1'] = x1_val
                        nd['y0'] = y0_val
                        nd['y1'] = y1_val
                        # Create output CSV for downstream aggregation
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path_to_new_file = 'project_MGNfelis_Oct2021_PatchesTraining.csv'
                        if os.path.exists(output_path_to_new_file):
                            new_df.to_csv(output_path_to_new_file, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path_to_new_file, header=True, index=False)
                    elif (cy >= 1100) and (cy <= 1548):
                        # Offset (as negative for second half of image dim)
                        # Calculate the residual (i.e. distance from closest threshold) to dynamically set the offset
                        prev_bound = 1099
                        res = cy - prev_bound
                        off_set = round(res*(0.25))
                        # Set the upper coordinate
                        cy0 = (cy - 200) - off_set
                        cy1 = (cy + 200) - off_set
                        """ OUTPUT """
                        # Create Nomenclature
                        # Format the z-value to have 100000s place (for downstream sorting on number)
                        x0_val = '{:0>6d}'.format(cx0)
                        y0_val = '{:0>6d}'.format(cy0)
                        x1_val = '{:0>6d}'.format(cx1)
                        y1_val = '{:0>6d}'.format(cy1)
                        ob_subarray = 'y' + str(y0_val) + '_' + 'x' + str(x0_val)
                        object_identity =  ob_subarray + '_' + omdata
                        filename = container_2D + '_' + object_identity + '.png'
                        # Establish the path for the image when output
                        out_npath = os.path.join(odir, filename)
                        # Select the array to be saved from data
                        subarray = p_mask[cy0:cy1, cx0:cx1]
                        # Convert the array to an image
                        im = Image.fromarray(subarray)
                        # Call the save image function
                        im.save(out_npath)
                        # Export metadata as csv for downstream processing
                        # First, get the name of the original reference file to search on...
                        # Then, record this file's identity and metadata of interest
                        nd = {}
                        nd['ref_key'] = container_2D
                        nd['mask_id'] = filename
                        nd['mask_path'] = out_npath
                        nd['ob_class'] = otype
                        nd['ob_user'] = user
                        nd['centroidX'] = cx
                        nd['centroidY'] = cy
                        nd['object_area'] = area
                        nd['object_radius'] = radius
                        nd['x0'] = x0_val
                        nd['x1'] = x1_val
                        nd['y0'] = y0_val
                        nd['y1'] = y1_val
                        # Create output CSV for downstream aggregation
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path_to_new_file = 'project_MGNfelis_Oct2021_PatchesTraining.csv'
                        if os.path.exists(output_path_to_new_file):
                            new_df.to_csv(output_path_to_new_file, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path_to_new_file, header=True, index=False)
                    elif (cy >= 1549) and (cy <= 1991):
                        # Offset (as negative for second half of image dim)
                        # Calculate the residual (i.e. distance from closest threshold) to dynamically set the offset
                        prev_bound = 1548
                        res = cy - prev_bound
                        off_set = round(res*(0.25))
                        # Set the upper coordinate
                        cy0 = (cy - 200) - off_set
                        cy1 = (cy + 200) - off_set
                        """ OUTPUT """
                        # Create Nomenclature
                        # Format the z-value to have 100000s place (for downstream sorting on number)
                        x0_val = '{:0>6d}'.format(cx0)
                        y0_val = '{:0>6d}'.format(cy0)
                        x1_val = '{:0>6d}'.format(cx1)
                        y1_val = '{:0>6d}'.format(cy1)
                        ob_subarray = 'y' + str(y0_val) + '_' + 'x' + str(x0_val)
                        object_identity =  ob_subarray + '_' + omdata
                        filename = container_2D + '_' + object_identity + '.png'
                        # Establish the path for the image when output
                        out_npath = os.path.join(odir, filename)
                        # Select the array to be saved from data
                        subarray = p_mask[cy0:cy1, cx0:cx1]
                        # Convert the array to an image
                        im = Image.fromarray(subarray)
                        # Call the save image function
                        im.save(out_npath)
                        # Export metadata as csv for downstream processing
                        # First, get the name of the original reference file to search on...
                        # Then, record this file's identity and metadata of interest
                        nd = {}
                        nd['ref_key'] = container_2D
                        nd['mask_id'] = filename
                        nd['mask_path'] = out_npath
                        nd['ob_class'] = otype
                        nd['ob_user'] = user
                        nd['centroidX'] = cx
                        nd['centroidY'] = cy
                        nd['object_area'] = area
                        nd['object_radius'] = radius
                        nd['x0'] = x0_val
                        nd['x1'] = x1_val
                        nd['y0'] = y0_val
                        nd['y1'] = y1_val
                        # Create output CSV for downstream aggregation
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path_to_new_file = 'project_MGNfelis_Oct2021_PatchesTraining.csv'
                        if os.path.exists(output_path_to_new_file):
                            new_df.to_csv(output_path_to_new_file, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path_to_new_file, header=True, index=False)
                    elif (cy >= 1992) and (cy <= 2192):
                        # No offset
                        cy0 = 1792
                        cy1 = 2192
                        """ OUTPUT """
                        # Create Nomenclature
                        # Format the z-value to have 100000s place (for downstream sorting on number)
                        x0_val = '{:0>6d}'.format(cx0)
                        y0_val = '{:0>6d}'.format(cy0)
                        x1_val = '{:0>6d}'.format(cx1)
                        y1_val = '{:0>6d}'.format(cy1)
                        ob_subarray = 'y' + str(y0_val) + '_' + 'x' + str(x0_val)
                        object_identity =  ob_subarray + '_' + omdata
                        filename = container_2D + '_' + object_identity + '.png'
                        # Establish the path for the image when output
                        out_npath = os.path.join(odir, filename)
                        # Select the array to be saved from data
                        subarray = p_mask[cy0:cy1, cx0:cx1]
                        # Convert the array to an image
                        im = Image.fromarray(subarray)
                        # Call the save image function
                        im.save(out_npath)
                        # Export metadata as csv for downstream processing
                        # First, get the name of the original reference file to search on...
                        # Then, record this file's identity and metadata of interest
                        nd = {}
                        nd['ref_key'] = container_2D
                        nd['mask_id'] = filename
                        nd['mask_path'] = out_npath
                        nd['ob_class'] = otype
                        nd['ob_user'] = user
                        nd['centroidX'] = cx
                        nd['centroidY'] = cy
                        nd['object_area'] = area
                        nd['object_radius'] = radius
                        nd['x0'] = x0_val
                        nd['x1'] = x1_val
                        nd['y0'] = y0_val
                        nd['y1'] = y1_val
                        # Create output CSV for downstream aggregation
                        new_df = pd.DataFrame().append(nd, ignore_index=True)
                        output_path_to_new_file = 'project_MGNfelis_Oct2021_PatchesTraining.csv'
                        if os.path.exists(output_path_to_new_file):
                            new_df.to_csv(output_path_to_new_file, mode='a', header=False, index=False)
                        else:
                            new_df.to_csv(output_path_to_new_file, header=True, index=False)
                    else:
                        print('wtf!!!', container, oid)
                        continue
        bar.next()
    bar.finish()




#################################################
#################################################



""" Read-in the data """
# df_rev = pd.read_csv('.csv')
df_rev = pd.read_csv('revised_training_2021.csv')


# Call the main function
make_patch_polygons(df_rev)