from configparser import ConfigParser
import sys
import os
import shutil
import subprocess
import glob

#this script created by merging the preprocess_CJS_unique_file.py and AreTomo_automated_CJS.py scripts
#to not run into issues with having python plugins installed, use the version of python installed in /sw/eman2/2.31_workstation/envs/tomo_preproc/bin/python
#example text to run this script (paste directly into terminal on beatrice2):/sw/eman2/2.31_workstation/envs/tomo_preproc/bin/python Motioncor2_AreTomo_pipeline_CJS.py /goliath/rawdata/Gouaux/matsui/Aya_GouauxLab_OHSU_CLO3-position4-halfmoon_20220419 /beatrice/scratch2/matsui/202204_CLO_hipp /rawdata/Gouaux/matsui/Aya_GouauxLab_OHSU_CLO3-position4-halfmoon_20220419/ref.mrc 11 2.06 001_-10.0.tif.mdoc 1200 4

#check that imod, motioncor2, and AreTomo modules are loaded
code = subprocess.run('which imod',shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
if code.returncode != 0 :
    sys.exit("imod not found, please load module, install it or add it to your path!")

code = subprocess.run('which motioncor2',shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
if code.returncode != 0 :
    sys.exit("motioncor2 not found, please load module, install it or add it to your path!")

code = subprocess.run('which AreTomo',shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
if code.returncode != 0 :
    sys.exit("AreTomo not found, please load module, install it or add it to your path!")

# parameters that are particular to the filename convention
Tomo_Number_Fields = 3 # how many fields separated by underscore that can uniquely identify a tomogram
Tilt_Order_Field = 4   # the number of field that describes the tilt order
Tilt_Angle_Field = 5   # the number of field that describe the tilt angle, sometimes the .mrc.mdoc needs to be removed
String_to_Remove = ".tif.mdoc" # The string needs to be removed first before the string splitting
Number_Skip_Lines = 3          # number of lines in the mrc.mdoc files to be skipped

# variables for MotionCor2 script
frames_folder = sys.argv[1]     # used for the movie_path
output_folder = sys.argv[2]     # used for generating the output
gain_ref = sys.argv[3]          # gain ref
pixel_dose = sys.argv[4]        # used for calculating tilt dose
pixel_size = sys.argv[5]        # used for calculating tilt dose
unique_string = sys.argv[6] # used for getting unique filename string
# variables for AreTomo script
VolZ = sys.argv[7]          # VolZ used for AreTomo
OutBin = sys.argv[8]        # OutBin used for AreTomo

Tomo_Unique_Str = "{}".format(unique_string) # String in the filename that uniquely identify a tomogram

#initialize list of directories that will be output from motioncor2 and input for AreTomo
AreTomo_directories = []
AreTomo_mrcs = []

# Change the working folder to the frames_folder
os.chdir(frames_folder)
# Get the list of tomograms
tomogram_list_all = ["_".join(file.split("_")[:Tomo_Number_Fields]) for file in glob.glob(f'*{Tomo_Unique_Str}*')]
tomogram_list = list(set(tomogram_list_all))
number_of_tomograms = len(tomogram_list)
print("Number of tomograms to be analyzed: ",number_of_tomograms)
print("Tomogram names: ", tomogram_list)

# The processing of a single tomogram
for tomogram in tomogram_list:
    ## Preparation for each tomogram
    #1.making folder
    processed_folder = os.path.join(output_folder,tomogram)
    outfile = tomogram
    # make directory only if it does not already exist (continue if it does)
    if not os.path.exists(processed_folder):
        os.mkdir(processed_folder)
    #add processed_folder to AreTomo_directories list for later processing
    AreTomo_directories.append(processed_folder)
    tomo_process_mrcs = os.path.join(output_folder,tomogram,tomogram)
    AreTomo_mrcs.append(tomo_process_mrcs)
    #2.Initialize newstack commands
    newstack_command = "newstack "
    newstack_command_DW = "newstack "
    #3.Open tilt file and dose file for writing
    tilt_file = open(f"{processed_folder}/{outfile}.rawtlt","w")
    dose_file = open(f"{processed_folder}/{outfile}.dose","w")
    
    # Actual processing
    movie_list = glob.glob(f'*{tomogram}*mdoc')
    min_order = min([int(file.replace(String_to_Remove,"").split("_")[Tilt_Order_Field-1]) for file in movie_list])
    # Sort the movie list
    tilt_angle_list = [float(file.replace(String_to_Remove,"").split("_")[Tilt_Angle_Field-1]) for file in movie_list]
    sorted_movie_list = [movie for _,movie in sorted(zip(tilt_angle_list,movie_list))]
    for movie in sorted_movie_list:
        tif_mdoc = ConfigParser()
        with open(movie) as stream:
            tif_mdoc.read_string(''.join(line for line in stream.readlines()[Number_Skip_Lines:]))
        section = list(tif_mdoc.keys())[-1]
        path_mdoc = tif_mdoc[section]['SubFramePath']
        newpath = "\\".join(path_mdoc.split("\\")[:-1])
        replacement = "\\" + movie.replace('.mdoc', '')
        print("Processing " + movie.replace('.mdoc', ''))
        path = newpath + replacement
        exp_time = float(tif_mdoc[section]['ExposureTime'])
        tilt = tif_mdoc[section]['TiltAngle']
        order = int(path.split('\\')[-1].split('_')[Tilt_Order_Field -1]) - min_order
        num_frames = int(tif_mdoc[section]['NumSubFrames']) 
        tilt_dose = float(pixel_dose) / (float(pixel_size) ** 2) * exp_time
        frame_dose = tilt_dose / num_frames
        prior_dose = tilt_dose * order
        movie_path = frames_folder + '/' + path.split('\\')[-1]
        if os.path.exists(f"{processed_folder}/tilt{order}_{tilt}.mrc"): # continue to next iteration of loop if micrograph already processed
            print("Already processed " + movie.replace('.mdoc', ''))
            continue
        command = f"motioncor2 -InTiff {movie_path} -OutMrc {processed_folder}/tilt{order}_{tilt}.mrc -Gpu 0 1 -UseGpus 1 -Gain {gain_ref} -Iter 10 -FtBin  1.963 -InitDose {prior_dose} -FmDose {frame_dose} -PixSize {float(pixel_size)/2} -kV 300 -Group 2 > {processed_folder}/tilt{order}_{tilt}.log" 
        os.system(command)
        newstack_command += f"{processed_folder}/tilt{order}_{tilt}.mrc "
        newstack_command_DW += f"{processed_folder}/tilt{order}_{tilt}_DW.mrc "
        tilt_file.write(f"{tilt}\n")
        dose_file.write(f"{prior_dose} {tilt_dose}\n")
    
    # Close tht tile file and dose file 
    tilt_file.close()
    dose_file.close()
    # Finalize the newstack commands
    newstack_command += f"{processed_folder}/{outfile}.mrc"
    newstack_command_DW += f"{processed_folder}/{outfile}_DW.mrc"
    # Run the newstack commands
    os.system(newstack_command)
    os.system(newstack_command_DW)
    # Delete tilt.mrc files
    subprocess.run(f"rm {processed_folder}/tilt*mrc", shell=True)

# This python script can process all the motion corrected tomograms after running the preprocess motioncor2 script

# Change the working folder to the output_folder
os.chdir(output_folder)

# Process all tomograms with AreTomo
for (tomogram, mrc) in zip(AreTomo_directories, AreTomo_mrcs):   
	try:
		os.chdir(tomogram)
	except:
		print(tomogram + " does not exist")
		continue
	else:
		command = f"AreTomo -InMrc {mrc}.mrc -OutMrc {mrc}_volume.mrc -VolZ {VolZ} -OutBin {OutBin} -AngFile *.rawtlt"
		os.system(command)
		os.chdir(output_folder)
		print(tomogram + " is done")

print("MotionCor2 and AreTomo pipeline is complete.")
