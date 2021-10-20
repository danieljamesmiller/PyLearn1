##  Running the 2d pipeline on nnUNet using the Graham cluster:

Here's some stuff

### Nice to Haves:
To get the number of files in a directory:
```
ls | wc -l
```

Printing the live number of files in a directory in a loop: (`Ctrl-c` to exit)

    while :; do ls | wc -l; done

To add group permissions to every file under (and including) a folder:
    
    Chmod g+wx folderName && Chmod g+wx folderName/*

### Prerequisites: 
- Tar file (from folder) of PNGs with RGB channels containing reference and label images for both training and testing where:
  - Reference image files end in: 	_refererence.png
    - ex. heres_a_file_name_reference.png
  - Label image files end in:		_Neuron.png
    - ex. heres_another_file_name_Neuron.png
- Start in the right directory:
    
        cd ~/projects/def-akhanf-ab/rohil/brainMap

### 1. Untar folder
Note:
-   `tarred_data.tar.gz` should be replaced with your compressed folder
-   `neuron_data` should be replaced with the folder you would like to untar the images into

Copy over the compressed folder into the current directory:
    # cp = copy; -r = recursive; destination dir; source dir (./ denotes this)
    cp -r ~/projects/ctb-akhanf/dmille95/preliminary/tarred_data.tar.gz ./

Create a directory to untar the data into and decompress it:

    # this is to decompress and move into new dir; 'neuron_data' is just a place holder here...
    mkdir neuron_data
    # -xf and -C commands are flags; these are likely to do the decompression; just stack these
    tar -xf tarred_data.tar.gz -C neuron_data/


### 2. Sort PNGs into train and test folders
nnUNET expects us to split data into the following directory structure:
    
    ./training
        input/
        output/

    ./testing
        input/
        output/


The script `data_sort.py` takes in 4 arguments:
  - train/test split (in range [0,1])
  - Suffix of reference image files
  - Suffix of label image files
  - Folder that data was untarred into

Run the script to do just that:
    
    cd ~/projects/def-akhanf-ab/rohil/brainMap/
    python3 data_sort.py 0.7 _reference _Neuron neuron_data

### 3. Convert PNGs to NIFTIs + Put in right folders for nnUNet
nnUNet expects 3 environment variables to be defined:

    # First snippet is a one-time upon set-up run....
    # one right arrow is to write, two right arrows is to append;
    Cat >> ~/.bashrc

    export RESULTS_FOLDER="~/projects/def-akhanf-ab/rohil/nnUNet/nnunet/dataset_conversion/nnUNet_trained_models"

    export nnUNet_preprocessed="~/projects/def-akhanf-ab/rohil/nnUNet/nnunet/dataset_conversion/nnUNet_preprocessed"

    export nnUNet_raw_data_base="~/projects/def-akhanf-ab/rohil/nnUNet/nnunet/dataset_conversion/nnUNet_raw"

    Ctrl-D

Now we source the ~/.bashrc file to load these changes:

    source ~/.bashrc



Now we run nnUNet's provided conversion script:
(NOTE: Once again, be sure to replace `neuron_data` with the folder you untarred into)
    
    # Note to change the task number
    # The task number MUST be greater than 100

    cd ~/projects/def-akhanf-ab/rohil/nnUNet/nnunet/dataset_conversion
    python3 png_to_nifti.py neuron_data Task102_BetterBrainData

After it finishes, confirm you can find the data in the following folder: 
NOTE: Be sure to change the final folder in the given query to the folder name provided to `png_to_nifty.py`.

    cd ~/projects/def-akhanf-ab/rohil/nnUNet/nnunet/dataset_conversion/nnUNet_raw/nnUNet_raw_data/Task102_BetterBrainMap

### 4. Preprocess the NIFTI data
Provision a GPU:

    cd ~/projects/def-akhanf-ab/rohil
    ./task120_interactive.sh

To run 2D preprocessing: 
    
    # -t (followed by task number)
    # ensure compatibility of this task number and the above....
    nnUNet_plan_and_preprocess -t 102 -pl3d None

Relinquish the GPU:

    Ctrl-D

### 5. Train (w/ 5-fold cross validation)


Train:

    # Do not train in an interactive session
    # sbatch allows monitoring remotely
    # make sure the task# matches the above...
    cd ~/projects/def-akhanf-ab/rohil   
    sbatch task102.sh
