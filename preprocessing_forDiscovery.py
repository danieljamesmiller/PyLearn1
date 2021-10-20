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
# from skimage.measure import find_contours
from skimage.morphology import disk, local_minima
# from skimage.segmentation import *
# from skimage.restoration import *
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


def make_preprocessing_inventory(top_dir):
	""" MODULE TO RECURSIVELY CATALOGUE DIRECTORY """
	# Set up output filename
	run_location = top_dir.split('/')[-1:][0]
	# The extension to search for
	extensions = ['.jpx', '.txt']
	run_time = datetime.now().strftime('%m-%d')
	# First, cataloge the directory of images and their 
	# paths (i.e. the file, and each one's location for retrieval)
	for dirpath, dirnames, files in os.walk(top_dir):
		ln = len(files)
		bar = IncrementalBar('Processing:', max=ln)
		for name in files:
			if name.lower().endswith(tuple(extensions)):
				item_path = os.path.join(dirpath, name)
				nd = {}
				nd['fpath'] = item_path
				nd['fname'] = name
				# Make new dataframe with output
				df1 = pd.DataFrame().append(nd, ignore_index=True)
				out_fname = run_location + '_Inventory.csv'
				if os.path.exists(out_fname):
					df1.to_csv(out_fname, mode='a', header=False, index=False)
				else:
					df1.to_csv(out_fname, header=True, index=False)
			bar.next()
		bar.finish()


##############################################


def transform_preprocessing_inventory(input_df):
	# Afterwards, read-in the processed csv
	mydf = pd.read_csv(input_df)
	mylist = sorted(mydf['fname'].unique().tolist())
	ln = len(mylist)
	bar = IncrementalBar('Processing:', max=ln)
	completed_list = []
	for file in mylist:
		if file not in completed_list:
			completed_list.append(file)
		else:
			continue
		# Loop through all the .jpx files, and for each one, grab the associated assembly data file
		# adding it to the output dataframe as a column or two...
		if '.jpx' in file:
			# First, check to ignore hidden or broken files
			if '._' in file:
				continue
			# Get metadata
			mdata = Path(file).stem
			# Grab the rows for the corresponding Assembly Data for its fpath
			container = '_'.join(mdata.split('_')[:-1])
			search_term = container + '_AssemblyData.txt'
			df_t1 = mydf.loc[mydf['fname'] == search_term].copy()
			if len(df_t1) < 1:
				continue
			assemble_fpath = df_t1['fpath'].values[0]
			# Grab the row for each file to get its fpath
			df_t0 = mydf.loc[mydf['fname'] == file].copy()
			if len(df_t0) < 1:
				continue
			image_fpath = df_t0['fpath'].values[0]
			# Set up output
			nd = {}
			nd['container'] = container
			nd['fname'] = file
			nd['fpath'] = image_fpath
			nd['reconstruct_fpath'] = assemble_fpath
			new_df = pd.DataFrame().append(nd, ignore_index=True)
			output_path = 'preprocessing_Inventory1.csv'
			if os.path.exists(output_path):
				new_df.to_csv(output_path, mode='a', header=False, index=False)
			else:
				new_df.to_csv(output_path, header=True, index=False)
		bar.next()
	bar.finish()


##############################################


def ingest_assembly_data(input_file):
	""" Part 1 of Module is to convert the Assembly Data from TXT to CSV """
	# The goal here is to make accessible all the data from the Assembly file
	txt_file = input_file
	mdata = Path(input_file).stem
	csv_file = mdata + '.csv'
	with open(txt_file, 'rt') as infile, open(csv_file, 'w+') as outfile:
			stripped = (line.strip() for line in infile)
			lines = (line.split(",") for line in stripped if line)
			writer = csv.writer(outfile)
			writer.writerows(lines)
	""" Part 2 of Module is to read in the converted original file & split it up """
	# Read in the converted original file
	df0 = pd.read_csv(csv_file, header=None)
	# Split into relevant parts
	# Constant size for end part of file data, as the last four rows are the footer (don't need)
	# Constant size for top part of file data, as the first 27 rows are the header (do need)
	df_metadata = df0[0:26]
	# Variable size of coordinate data part of file data (i.e. outside header + footer; do need)
	df_data = df0[27:-4]
	# First, work with the metadata bc we need to bring it along with coordinate data
	# First, get the container metadata
	mdat = mdata.split('_')[:-1]
	mdt = '_'.join(mdat)
	modified_csv_fname = mdt + '_AssemblyModified.csv'
	# Obtain relevant information from metadata portion of file
	tile_dat = df_metadata[0][3]
	tile_num = tile_dat.split('=')[1]
	grx_dat = df_metadata[0][4]
	gridx = grx_dat.split('=')[1]
	gry_dat = df_metadata[0][5]
	gridy = gry_dat.split('=')[1]
	# Transform converted file of tile location data
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
				# Adjust tile number by 1 bc of indexing,
				# from 0 but image naming starts at 1...
				tile_no = int(tile_var[4:]) + 1
				tile_num = '{:0>6d}'.format(tile_no)
				# tile_num = '{:0>6d}'.format(tile_no)
				coord_val = '{:0>6d}'.format(value)
				tile_id = mdt + '_' + tile_num
				# Set up naming for output
				nd = {}
				nd['tile_id'] = tile_id
				nd['coord_label'] = coord_label
				nd['coord_value'] = coord_val
				nd['grid_size'] = tile_num
				nd['gridX'] = gridx
				nd['gridY'] = gridy
				new_df = pd.DataFrame().append(nd, ignore_index=True)
				output_path = mdt + '_AssemblyModified.csv'
				if os.path.exists(output_path):
					new_df.to_csv(output_path, mode='a', header=False, index=False)
				else:
					new_df.to_csv(output_path, header=True, index=False)
	""" Part 3 of Module: 2nd transform & output_DF to name each container of images """
	# Read in the modified data
	df1 = pd.read_csv(modified_csv_fname)
	# Make unique list of objects
	flist = sorted(df1['tile_id'].unique().tolist())
	# Loop through unique list
	comp_list = []
	for item in flist:
		if item not in comp_list:
			comp_list.append(item)
		else:
			continue
		# Grab relevant data from original dataframe
		df_temp = df1.loc[df1['tile_id'] == item]
		# And loop through to create new file
		for index, row in df_temp.iterrows():
			if 'X' in row.coord_label:
				x_coord = row.coord_value
				x_coord = '{:0>6d}'.format(x_coord)
			if 'Y' in row.coord_label:
				y_coord = row.coord_value
				y_coord = '{:0>6d}'.format(y_coord)
		# For april 27th, keep the naming per old school 
		# the processing of cases W310 and W312
		tile_id = row.tile_id
		mdata = tile_id.split('_')
		mdat = mdata[:-1]
		sample = mdat[0]
		series = mdat[1]
		z_coord = mdat[2]
		block = mdat[3]
		side = mdat[4]
		x_coord = 'x' + x_coord
		y_coord = 'y' + y_coord
		grid_size = row.grid_size
		dimX = row.gridX
		dimY = row.gridY
		identity = '_'.join([sample, side, series, block, z_coord, y_coord, x_coord])
		# Output new dataframe
		nd = {}
		nd['TileID_Local'] = tile_id
		nd['TileID_Global'] = identity
		nd['Grid_Size'] = grid_size
		nd['dimY'] = dimY
		nd['dimX'] = dimX
		new_df = pd.DataFrame().append(nd, ignore_index=True)
		output_path = mdt + '_AssemblyPrep.csv'
		if os.path.exists(output_path):
			new_df.to_csv(output_path, mode='a', header=False, index=False)
		else:
			new_df.to_csv(output_path, header=True, index=False)

# # Read-in the finally prepared file to ensure function performance
# df2 = pd.read_csv('W310_SeriesBDAd_Section0002_Cortex_Left_AssemblyPrep.csv')
# print(len(df2))

################################

########################
## Start Object Chunking
########################
def chunk_it_move_it_from(dataframe1, dataframe2):
	# Set the output folder for generated subarrays
	curr_dir = os.getcwd()
	out_dirpath = curr_dir + '/subarrays/'
	# Read-in the dataset
	df0 = pd.read_csv(dataframe1)
	df1 = pd.read_csv(dataframe2)
	# df0 = pd.read_csv('W310_SeriesBDAd_Section0002_Cortex_Left_AssemblyPrep.csv')
	ln = len(df0)
	bar = IncrementalBar('Processing:', max=ln)
	# Loop through prepared dataframe
	for index, row in df0.iterrows():
		# Settle nomenclature for search from dataframe
		file_to_chunk = row.TileID_Local + '.jpx'
		temp_search = df1.loc[df1['fname'] == file_to_chunk].copy()
		chunk_path = temp_search['fpath'].values[0]
		# Set up nomenclature for subarray output
		mdata = row.TileID_Global.split('_')
		metadata = '_'.join(mdata[:-2])
		x_coord = int(mdata[-1][1:])
		y_coord = int(mdata[-2][1:])
		# Read-in the file, after setting nomenclature
		img0 = imageio.imread(chunk_path)
		# Get dim
		xdim = img0.shape[1]
		ydim = img0.shape[0]
		# Derive limits
		xlim = round(xdim/200)
		ylim = round(ydim/200)
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
				right = 200
				for y in range(0, ylim, 1):
					if y == 0:
						upper = 0
						Global_Upper = upper + y_coord
						lower = 200
						# Set coordinate values to being larger bc of global coordinate space
						GlobalX = 'x' + '{:0>6d}'.format(Global_Left)
						GlobalY = 'y' + '{:0>6d}'.format(Global_Upper)
						out_fname = metadata + '_' + GlobalY + '_' + GlobalX + '_reference.png'
						# Write File 1
						subarray = img0[upper:lower, left:right]
						im1 = Image.fromarray(subarray)
						im1.save(out_fname)
						# Move file to directory FOR_DISCOVERY
						src_path = os.path.join(curr_dir, out_fname)
						shutil.move(src=src_path, dst=out_dirpath)
					elif (y > 0 and y < ynum):
						upper = y*200
						Global_Upper = upper + y_coord
						lower = upper + 200
						# Set coordinate values to being larger bc of global coordinate space
						GlobalX = 'x' + '{:0>6d}'.format(Global_Left)
						GlobalY = 'y' + '{:0>6d}'.format(Global_Upper)
						out_fname = metadata + '_' + GlobalY + '_' + GlobalX + '_reference.png'
						# Write File 2
						subarray = img0[upper:lower, left:right]
						im1 = Image.fromarray(subarray)
						im1.save(out_fname)
						# Move file to directory FOR_DISCOVERY
						src_path = os.path.join(curr_dir, out_fname)
						shutil.move(src=src_path, dst=out_dirpath)
					elif y == ynum:
						upper = 1992
						Global_Upper = upper + y_coord
						lower = 2192
						# Set coordinate values to being larger bc of global coordinate space
						GlobalX = 'x' + '{:0>6d}'.format(Global_Left)
						GlobalY = 'y' + '{:0>6d}'.format(Global_Upper)
						out_fname = metadata + '_' + GlobalY + '_' + GlobalX + '_reference.png'
						# Write File 3
						subarray = img0[upper:lower, left:right]
						im1 = Image.fromarray(subarray)
						im1.save(out_fname)
						# Move file to directory FOR_DISCOVERY
						src_path = os.path.join(curr_dir, out_fname)
						shutil.move(src=src_path, dst=out_dirpath)
					else:
						continue
			# Second option
			elif (x > 0 and x < xnum):
				left = x*200
				Global_Left = left + x_coord
				right = left + 200
				for y in range(0, ylim, 1):
					if y == 0:
						upper = 0
						Global_Upper = upper + y_coord
						lower = 200
						# Set coordinate values to being larger bc of global coordinate space
						GlobalX = 'x' + '{:0>6d}'.format(Global_Left)
						GlobalY = 'y' + '{:0>6d}'.format(Global_Upper)
						out_fname = metadata + '_' + GlobalY + '_' + GlobalX + '_reference.png'
						# Write File 4
						subarray = img0[upper:lower, left:right]
						im1 = Image.fromarray(subarray)
						im1.save(out_fname)
						# Move file to directory FOR_DISCOVERY
						src_path = os.path.join(curr_dir, out_fname)
						shutil.move(src=src_path, dst=out_dirpath)
					elif (y > 0 and y < ynum):
						upper = y*200
						Global_Upper = upper + y_coord
						lower = upper + 200
						# Set coordinate values to being larger bc of global coordinate space
						GlobalX = 'x' + '{:0>6d}'.format(Global_Left)
						GlobalY = 'y' + '{:0>6d}'.format(Global_Upper)
						out_fname = metadata + '_' + GlobalY + '_' + GlobalX + '_reference.png'
						# Write File 5
						subarray = img0[upper:lower, left:right]
						im1 = Image.fromarray(subarray)
						im1.save(out_fname)
						# Move file to directory FOR_DISCOVERY
						src_path = os.path.join(curr_dir, out_fname)
						shutil.move(src=src_path, dst=out_dirpath)
					elif y == ynum:
						upper = 1992
						Global_Upper = upper + y_coord
						lower = 2192
						# Set coordinate values to being larger bc of global coordinate space
						GlobalX = 'x' + '{:0>6d}'.format(Global_Left)
						GlobalY = 'y' + '{:0>6d}'.format(Global_Upper)
						out_fname = metadata + '_' + GlobalY + '_' + GlobalX + '_reference.png'
						# Write File 6
						subarray = img0[upper:lower, left:right]
						im1 = Image.fromarray(subarray)
						im1.save(out_fname)
						# Move file to directory FOR_DISCOVERY
						src_path = os.path.join(curr_dir, out_fname)
						shutil.move(src=src_path, dst=out_dirpath)
					else:
						continue
			# Third option
			elif x == xnum:
				left = 2552
				Global_Left = left + x_coord
				right = 2752
				for y in range(0, ylim, 1):
					if y == 0:
						upper = 0
						Global_Upper = upper + y_coord
						lower = 200
						# Set coordinate values to being larger bc of global coordinate space
						GlobalX = 'x' + '{:0>6d}'.format(Global_Left)
						GlobalY = 'y' + '{:0>6d}'.format(Global_Upper)
						out_fname = metadata + '_' + GlobalY + '_' + GlobalX + '_reference.png'
						# Write File 7
						subarray = img0[upper:lower, left:right]
						im1 = Image.fromarray(subarray)
						im1.save(out_fname)
						# Move file to directory FOR_DISCOVERY
						src_path = os.path.join(curr_dir, out_fname)
						shutil.move(src=src_path, dst=out_dirpath)
					elif (y > 0 and y < ynum):
						upper = y*200
						Global_Upper = upper + y_coord
						lower = upper + 200
						# Set coordinate values to being larger bc of global coordinate space
						GlobalX = 'x' + '{:0>6d}'.format(Global_Left)
						GlobalY = 'y' + '{:0>6d}'.format(Global_Upper)
						out_fname = metadata + '_' + GlobalY + '_' + GlobalX + '_reference.png'
						# Write File 8
						subarray = img0[upper:lower, left:right]
						im1 = Image.fromarray(subarray)
						im1.save(out_fname)
						# Move file to directory FOR_DISCOVERY
						src_path = os.path.join(curr_dir, out_fname)
						shutil.move(src=src_path, dst=out_dirpath)
					elif y == ynum:
						upper = 1992
						Global_Upper = upper + y_coord
						lower = 2192
						# Set coordinate values to being larger bc of global coordinate space
						GlobalX = 'x' + '{:0>6d}'.format(Global_Left)
						GlobalY = 'y' + '{:0>6d}'.format(Global_Upper)
						out_fname = metadata + '_' + GlobalY + '_' + GlobalX + '_reference.png'
						# Write File 9
						subarray = img0[upper:lower, left:right]
						im1 = Image.fromarray(subarray)
						im1.save(out_fname)
						# Move file to directory FOR_DISCOVERY
						src_path = os.path.join(curr_dir, out_fname)
						shutil.move(src=src_path, dst=out_dirpath)
					else:
						continue
			else:
				continue
		bar.next()
	bar.finish()



##############################################



# def image_chunker_and_mover(image):
# curr_dir = os.getcwd()
# # For segmenting stained (black) neurons in light brown tissue -- MGN project
# im_rgb = imageio.imread(image)
# metadata = Path(image).stem
# mdata = '_'.join(metadata.split('_')[:-1])
# # Set nomenclature
# out_fname = mdata + '_reference.png' 
# out_dir = '/Volumes/ML1/deep_learning/preprocessing/subarrays/'
# im0 = Image.fromarray(pred)
# im0.save(out_fname)
# # Now, move both files to folder containing target segmentations
# shutil.move(src=out_fname, dst=out_dir)
# shutil.move(src=image, dst=out_dir)


##################


def make_subarrays_from_directory_loop(input_dataframe):
	dfG = pd.read_csv(input_dataframe)
	container_list = sorted(dfG['container'].unique().tolist())
	ln = len(dfG)
	bar = IncrementalBar('Processing:', max=ln)
	completed_containers = []
	for container in container_list:
		if container not in completed_containers:
			completed_containers.append(container)
		else:
			continue
		# call function to transform the Assembly Data file
		# which outputs the complete list of subarrays that
		# need to be created...
		temp0 = dfG.loc[dfG['container']==container].copy()
		container_assembly_data_fpath = temp0['reconstruct_fpath'].values[0]
		ingest_assembly_data(container_assembly_data_fpath)
		# so that we can then just run it on a dataframe
		# that contains all the relevant information about
		# where each file will be that needs to be chunked
		prep_container_assembly_fpath = container + '_AssemblyPrep.csv'
		chunk_it_move_it_from(prep_container_assembly_fpath, input_dataframe)
		bar.next()
	bar.finish()




""" Outline for Pipeline  """
# # First, run the catalogue step ---
# # Set the top argument for walk as current directory
# cdir = os.getcwd()
# make_preprocessing_inventory(cdir)
# # Then, run the transformation step
# transform_preprocessing_inventory('preprocessing_Inventory.csv')
# Finally, call the function to make and move subarrays to final resting place
# ... prior to being uploaded to graham
make_subarrays_from_directory_loop('preprocessing_Inventory1.csv')




