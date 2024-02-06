# /// script
# dependencies = ["mdocfile","typer","rich","pydantic"]
# ///
import sys
import os
import subprocess
from pathlib import Path
import typer
from rich import print
from pydantic import BaseModel
from typing import Optional, List
from enum import Enum
from mdocfile.data_models import Mdoc
from rich.progress import track

# Create Enum for the different types of tomograms
class TomoType(str, Enum):
    tif = "tif"
    eer = "eer"

class MovieInfo(BaseModel):
    FileName: Path
    TiltAngle: float
    PixelSize: float
    ExposureOrder: int

class TomogramInfo(BaseModel):
    TomoName: str
    PixelSize: Optional[float] = None
    Type: Optional[TomoType] = None
    GainRef: Optional[str] = None




# Run this script using pipx run Motioncor3_AreTomo_JE.py /path_to_folders_with_frames
# The script will process all the tomograms in the folder and output the motion corrected tomograms
def main(
    frames_folder: Path = typer.Argument(..., help="Path to the folder with frames"),
    tiltseries_folder: Path = typer.Argument(..., help="Path to the folder where tiltseries will be stored"),
    pixel_size: float = typer.Argument(..., help="Pixel size of movies in Angstroms"),
    run_aretomo: bool = typer.Option(False, help="Run AreTomo after MotionCor3"),
    dose_per_tilt: float = typer.Option(4.55, help="Dose per tilt in e/A^2"),
    gpu: float = typer.Option(0, help="GPU ID for MotionCor"),

):
    #check that IMOD, MotionCor3, and AreTomo modules are loaded
    code = subprocess.run('which imod',shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    if code.returncode != 0 :
        sys.exit("IMOD not found, please load module, install it or add it to your path!")

    code = subprocess.run('which MotionCor3',shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    if code.returncode != 0 :
        sys.exit("MotionCor3 not found, please load module, install it or add it to your path!")

    if run_aretomo:
        code = subprocess.run('which AreTomo',shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        if code.returncode != 0 :
            sys.exit("AreTomo not found, please load module, install it or add it to your path!")

    #initialize list of directories that will be output from motioncor3 and input for AreTomo2
    AreTomo_directories = []
    AreTomo_mrcs = []
    
    # Get the list of tomograms
    tomograms = [folder.name for folder in frames_folder.glob("*")]
    tomograms = [TomogramInfo(TomoName=tomogram) for tomogram in tomograms]
    
    # The processing of a single tomogram
    for tomogram in tomograms:
        tiltseries_file = tiltseries_folder / tomogram.TomoName / (tomogram.TomoName + '.mrc')
        tiltseries_evn_file = tiltseries_folder / tomogram.TomoName / (tomogram.TomoName + '_EVN.mrc')
        tiltseries_odd_file = tiltseries_folder / tomogram.TomoName / (tomogram.TomoName + '_ODD.mrc')
        rawtlt_file = tiltseries_folder / tomogram.TomoName / (tomogram.TomoName + '.rawtlt')
        dose_file = tiltseries_folder / tomogram.TomoName / (tomogram.TomoName + '.dose')
        csv_file = tiltseries_folder / tomogram.TomoName / (tomogram.TomoName + '.csv')
        
        #add tomogram to list for AreTomo processing
        if run_aretomo:
            tiltseries_mrc = tiltseries_folder / tomogram.TomoName / tomogram.TomoName
            processed_folder = os.path.join(tiltseries_folder,tomogram.TomoName)
            AreTomo_directories.append(processed_folder)
            AreTomo_mrcs.append(tiltseries_mrc)
        
        if tiltseries_file.exists() and tiltseries_evn_file.exists() and tiltseries_odd_file.exists() and rawtlt_file.exists() and dose_file.exists() and csv_file.exists():
            print(f"Skipping {tomogram.TomoName}")
            continue

        movie_path = frames_folder / tomogram.TomoName
        # Test if eer or tif and generate movie list
        tif_list = list(movie_path.glob(f"{tomogram.TomoName}*.tif"))
        eer_list = list(movie_path.glob(f"{tomogram.TomoName}*.eer"))
        if len(tif_list) > 0:
            tomogram.Type = TomoType.tif
            movie_list = tif_list
            print(f"tif frames found for {tomogram.TomoName}")
        elif len(eer_list) > 0:
            tomogram.Type = TomoType.eer
            movie_list = eer_list
            print(f"eer frames found for {tomogram.TomoName}")
        else:
            print(f"No frames found for {tomogram.TomoName}")
            continue
        # Find gain reference
        if tomogram.Type == TomoType.tif:
            tomogram.GainRef = Path(movie_path / f"gain_reference.mrc")
            if not tomogram.GainRef.exists():
                print(f"Gain reference not found for {tomogram.TomoName}")
                continue
        elif tomogram.Type == TomoType.eer:
            potential_gain_ref = list(movie_path.glob(f"gain_reference.gain"))
            if len(potential_gain_ref) == 1:
                tomogram.GainRef = Path(potential_gain_ref[0])
            else:
                print(f"Gain reference not found for {tomogram.TomoName}")
                continue

        mc_outpath = frames_folder / tomogram.TomoName / "mc"
        mc_outpath.mkdir(exist_ok=True)
        # Get info for individual movies
        movie_info_list = []
        print(f"Motioncorrection {tomogram.TomoName}")
        for i,movie in track(enumerate(sorted(movie_list)), total=len(movie_list)):
            mdocinfo = Mdoc.from_file(str(movie)+'.mdoc').section_data[0]
            movieinfo = MovieInfo(
                FileName=movie,
                TiltAngle=mdocinfo.TiltAngle,
                ExposureOrder=i+1,
                PixelSize=pixel_size
            )
            movie_info_list.append(movieinfo)
            

            # Run Motioncor3
            command = f"MotionCor3 "
            if tomogram.Type == TomoType.tif:
                command += f" -InTiff {movieinfo.FileName} " 
            elif tomogram.Type == TomoType.eer:
                command += f" -InEer {movieinfo.FileName} –EerSampling 1 "
            command += f" -OutMrc {mc_outpath / ('tilt'+str(movieinfo.ExposureOrder) +'_' + str(round(movieinfo.TiltAngle)) +'.mrc')} "
            command += f" -FtBin {2.5 / movieinfo.PixelSize}"
            command += f" -Gpu {gpu}  -SplitSum 1 -UseGpus {gpu} -Gain {tomogram.GainRef}  -Group 2 "
            command += f"> {mc_outpath / ('tilt'+str(movieinfo.ExposureOrder) +'_' + str(round(movieinfo.TiltAngle)) +'.log')}" 
            print(command)
            subprocess.run(command, shell=True)
            
        # Newstack for all,evn,odd
            
        command = f"newstack "
        command_EVN = f"newstack "
        command_ODD = f"newstack "
        for movie in sorted(movie_info_list, key=lambda x: x.TiltAngle):
            command += f"{mc_outpath / ('tilt'+str(movie.ExposureOrder) +'_' + str(round(movie.TiltAngle)) +'.mrc')} "
            command_EVN += f"{mc_outpath / ('tilt'+str(movie.ExposureOrder) +'_' + str(round(movie.TiltAngle)) +'_EVN.mrc')} "
            command_ODD += f"{mc_outpath / ('tilt'+str(movie.ExposureOrder) +'_' + str(round(movie.TiltAngle)) +'_ODD.mrc')} "
        (tiltseries_folder / tomogram.TomoName).mkdir(exist_ok=True,parents=True)
        command += str(tiltseries_file)
        command_EVN += str(tiltseries_evn_file)
        command_ODD += str(tiltseries_odd_file)
        print(command)
        subprocess.run(command, shell=True)
        print(command_EVN)
        subprocess.run(command_EVN, shell=True)
        print(command_ODD)
        subprocess.run(command_ODD, shell=True)

        subprocess.run(f"rm {mc_outpath}/*", shell=True)

        TomogramInfo.PixelSize = movie_info_list[0].PixelSize
        with open(rawtlt_file, 'w') as f:
            for movie in sorted(movie_info_list, key=lambda x: x.TiltAngle):
                f.write(f"{movie.TiltAngle}\n")

        with open(dose_file, 'w') as f:
            for movie in sorted(movie_info_list, key=lambda x: x.TiltAngle):
                f.write(f"{dose_per_tilt * (movie.ExposureOrder-1)} {dose_per_tilt}\n")
        
        
        with open(csv_file, 'w') as f:
            for movie in sorted(movie_info_list, key=lambda x: x.ExposureOrder):
                f.write(f"{movie.ExposureOrder},{round(movie.TiltAngle)}\n")

        print("MotionCor3 processing is complete.")
        
        
    if run_aretomo:
        
        # Change the working directory to the tiltseries_folder
        os.chdir(tiltseries_folder)

        # Process all tomograms with AreTomo, if not already processed
        for (tomogram, mrc) in zip(AreTomo_directories, AreTomo_mrcs):   
            try:
                os.chdir(tomogram)
            except:
                print(tomogram + " does not exist - skipping.")
                continue
            AreTomo_output = f"{mrc}_volume_AreTomo.mrc"
            if os.path.exists(AreTomo_output):
                print(tomogram + " already processed - skipping.")
                continue
            else:
                print(f"Processing {mrc} with AreTomo.")
                command = f"AreTomo -InMrc {mrc}.mrc -OutMrc {mrc}_volume_AreTomo.mrc -VolZ 1200 -OutBin 4 -AngFile *.rawtlt -Gpu {gpu}"
                os.system(command)
                os.chdir(tiltseries_folder)
                print(tomogram + " is done.")

        print("AreTomo processing is complete.")

 

if __name__ == "__main__":
    typer.run(main)