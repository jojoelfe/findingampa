# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "watchfiles",
#     "feud[all]",
#    "findingampa @ /home/gouauxlab/andecath/findingampa",
# ]
# ///
import multiprocessing

import asyncio
from watchfiles import awatch
import feud
from pathlib import Path
import os


state = {
    "copy_files": {
    },
    "ddw": {
    },
    "previews": {
    },
    "membrainseg": {
    },
    "membrainseg_mesh": {  
    },
    "synapsenet": {
    },
    "tardis_mt": {
    },
    "aunp_match": {

    }
}

gpu_lock = asyncio.Lock()

async def copy_files(
        mdoc_done_path: str,
        destination: Path, 
        sample_name: str
):
    
    suffix_to_copy = [
        "_Vol.mrc",
        "_ODD_Vol.mrc",
        "_EVN_Vol.mrc",
    ]
    while True:
        with open(mdoc_done_path, "r") as f:
            for mdoc in f.readlines():
                mdoc = mdoc.strip()
                prefix = mdoc.split(".")[0]
                if prefix not in state["copy_files"]:
                # Copy the mdoc file to the destination folder
                    for suffix in suffix_to_copy:
                        success = False
                        final_path = destination / f"{sample_name}_{prefix}" / "aretomo" / f"{sample_name}_{prefix}{suffix}"
                        if not final_path.exists():                        
                            final_path.parent.mkdir(parents=True, exist_ok=True)
                            try:
                                print(f"Copy {(prefix + suffix)}")
                                # Copy the file
                                cpp = await asyncio.create_subprocess_shell(f"cp {mdoc_done_path.parent / (prefix + suffix)} {final_path}")
                                await cpp.wait()
                            except FileNotFoundError:
                                print(f"Failed to copy {prefix}{suffix}")                    
                    state["copy_files"][prefix] = True
        await asyncio.sleep(1)



async def ddw_aretomo(
    folder: Path = None
):
    import mrcfile
    try:
        if folder is None:
            folder = Path.cwd()
        tomo_name = list(folder.glob("*EVN_Vol.mrc"))[0].stem[:-8]
    except IndexError:
        return

    from subprocess import Popen, PIPE, DEVNULL, STDOUT

    print(f"Running DDW for {tomo_name}")
    p = await asyncio.create_subprocess_shell(f"scp *EVN_Vol.mrc *ODD_Vol.mrc 10.137.46.10:/juliet/scratch/andecath/ddw_temp/", shell=True, cwd=folder,stdout=DEVNULL,stderr=STDOUT)
    await p.wait()
    p = await asyncio.create_subprocess_shell(f"ssh 10.137.46.10 newstack -mu 1000000,0 /juliet/scratch/andecath/ddw_temp/{tomo_name}_EVN_Vol.mrc /juliet/scratch/andecath/ddw_temp/{tomo_name}_EVN_Vol.mrc ", shell=True,stdout=DEVNULL,stderr=STDOUT)
    await p.wait()
    p = await asyncio.create_subprocess_shell(f"ssh 10.137.46.10 newstack -mu 1000000,0 /juliet/scratch/andecath/ddw_temp/{tomo_name}_ODD_Vol.mrc /juliet/scratch/andecath/ddw_temp/{tomo_name}_ODD_Vol.mrc ", shell=True,stdout=DEVNULL,stderr=STDOUT)
    await p.wait()
    p = await asyncio.create_subprocess_shell(f"ssh 10.137.46.10 /juliet/scratch/andecath/DeepDeWedge/env/bin/ddw refine-tomogram --tomo0-files /juliet/scratch/andecath/ddw_temp/{tomo_name}_EVN_Vol.mrc  --tomo1-files /juliet/scratch/andecath/ddw_temp/{tomo_name}_ODD_Vol.mrc --subtomo-size 96 --mw-angle 84 --subtomo-overlap 32 --batch-size 10 --model-checkpoint-file /juliet/scratch/andecath/ddw_temp/epoch=409-val_loss=13.62301.ckpt --output-dir /juliet/scratch/andecath/ddw_temp/ --num-workers 10 --gpu 0", shell=True,stdout=DEVNULL,stderr=STDOUT)
    await p.wait()
    p = await asyncio.create_subprocess_shell(f"scp 10.137.46.10:/juliet/scratch/andecath/ddw_temp/{tomo_name}*_refined.mrc {tomo_name}_Vol_ddw.mrc", shell=True,cwd=folder,stdout=DEVNULL,stderr=STDOUT)
    await p.wait()
    print(f"Finished DDW {tomo_name}")

async def ddw(
        mdoc_done_path: str,
        destination: Path, 
        sample_name: str
):
    while True:
        with open(mdoc_done_path, "r") as f:
            for mdoc in f.readlines():
                mdoc = mdoc.strip()
                prefix = mdoc.split(".")[0]
                if prefix not in state["ddw"] and prefix in state["copy_files"]:
                
                    final_path = destination / f"{sample_name}_{prefix}" / "aretomo" / f"{sample_name}_{prefix}_Vol_ddw.mrc"

                    if not final_path.exists():
                        print(f"Running ddw for {prefix}")
                        # Run dw_aretomo in a thread
                        await ddw_aretomo(destination / f"{sample_name}_{prefix}" / "aretomo" )
                                    
                    state["ddw"][prefix] = True
        await asyncio.sleep(1)


async def create_previews(
        mdoc_done_path: Path,
        destination: Path,
        sample_name: str
):
    from subprocess import Popen, PIPE, DEVNULL, STDOUT

    while True:
        with open(mdoc_done_path, "r") as f:
            for mdoc in f.readlines():
                mdoc = mdoc.strip()
                prefix = mdoc.split(".")[0]
                if prefix not in state["previews"] and prefix in state["ddw"]:
                    
                    ddw_path = destination / f"{sample_name}_{prefix}" / "aretomo" / f"{sample_name}_{prefix}_Vol_ddw.mrc"
                    png_path = destination / f"{sample_name}_{prefix}" / "aretomo" / "pngs"
                    # Check how many pngs in png_path
                    if  png_path.exists() and len(list(png_path.glob("*.png"))) > 2:
                        state["previews"][prefix] = True
                        continue
                    print(f"Creating previews for {prefix}")
                    png_path.mkdir(exist_ok=True)
                    # Create previews
                    for slice in [120,150,180]:
                        p = await asyncio.create_subprocess_shell(
                            " ".join([
                                "mrc2tif",
                                "-p",
                                "-z",
                                f"{slice},{slice}",
                                "-a",
                                "0,0",
                                f"{ddw_path}",
                                f"{png_path}/{sample_name}_{prefix}_center"
                            ])
                        )
                        await p.wait()
                    state["previews"][prefix] = True
        await asyncio.sleep(10)

def run_membrainseg(
    folder: Path = None,
    tomo_name: str = None
):
    from findingampa.utils.assets import FA_ASSETS
    import mrcfile
    
    MEMBRAIN_MODEL_PATH = FA_ASSETS / "membrain" / "MemBrain_seg_v10_alpha.ckpt"

    from membrain_seg.segmentation.segment import segment
    import torch

    membrain_output_path = folder/"membrain"
    membrain_output_path.mkdir(parents=True, exist_ok=True)
    
    segmentation_file = segment(
        tomogram_path=str(folder / f"{tomo_name}_Vol_ddw.mrc"),
        ckpt_path=MEMBRAIN_MODEL_PATH,
        out_folder=membrain_output_path,
        segmentation_threshold=0.0,
        store_probabilities=True
    )
    # Make torch clear gpu memory
    torch.cuda.empty_cache()

async def membrainseg(
        mdoc_done_path: Path,
        destination: Path, 
        sample_name: str
):
    from concurrent.futures import ProcessPoolExecutor 

    loop = asyncio.get_event_loop()
    executor = ProcessPoolExecutor(max_workers=1)
    while True:
        with open(mdoc_done_path, "r") as f:
            for mdoc in f.readlines():
                mdoc = mdoc.strip()
                prefix = mdoc.split(".")[0]
                if prefix in state["ddw"] and prefix not in state["membrainseg"]:
                    membrain_folder = destination / f"{sample_name}_{prefix}" / "aretomo" / "membrain"
                    scores_glob = membrain_folder.glob("*scores.mrc")
                    if len(list(scores_glob)) == 0:
                        print(f"Running membrainseg for {prefix}")
                        async with gpu_lock:
                            await loop.run_in_executor(executor,run_membrainseg,destination / f"{sample_name}_{prefix}" / "aretomo", f"{sample_name}_{prefix}")
                    state["membrainseg"][prefix] = True
        await asyncio.sleep(1)

def run_aunp_match(
    folder: Path = None,
    tomo_name: str = None
):
    import torch
    import mrcfile
    from findingampa.utils.tm3d import calc_3dtm, fit_gaussian_3d_list
    from findingampa.utils.analysis import find_surfaces
    import starfile
    import pandas as pd 

    locations_tomo, locations_score_volume, score_volume = calc_3dtm(folder / f"{tomo_name}_Vol.mrc", high_pass=None)
    score_volume = torch.tensor(score_volume).cuda()
    fitted_params_list = fit_gaussian_3d_list(torch.tensor(score_volume), locations_score_volume.cuda(),12)
    offsets = torch.tensor([[f['x0'], f['y0'], f['z0']] for f in fitted_params_list])
    with mrcfile.open(folder / f"{tomo_name}_Vol.mrc", permissive=True) as f:
        tomo_tensor = torch.tensor(f.data).cuda()
        tomo_tensor /= tomo_tensor.std()

    tomo_fitted_params_list = fit_gaussian_3d_list(tomo_tensor, locations_tomo.int().cuda(), 8, sign=-1)

    d1, d2 = find_surfaces(locations_tomo.cpu().numpy())

    data = pd.DataFrame({'faTM_PeakX':locations_tomo[:,2].cpu().numpy(),
                        'faTM_PeakY':locations_tomo[:,1].cpu().numpy(),
                        'faTM_PeakZ':locations_tomo[:,0].cpu().numpy(),
                        'faTM_PeakScore':[score_volume[x,y,z].item() for x,y,z in locations_score_volume],
                        'faTM_gaussianFitX':offsets[:,0].cpu().numpy(),
                        'faTM_gaussianFitY':offsets[:,1].cpu().numpy(),
                        'faTM_gaussianFitZ':offsets[:,2].cpu().numpy(),
                        'faTM_gaussianFitSigmaX':[f['sigma_x'] for f in fitted_params_list],
                        'faTM_gaussianFitSigmaY':[f['sigma_y'] for f in fitted_params_list],
                        'faTM_gaussianFitSigmaZ':[f['sigma_z'] for f in fitted_params_list],
                        'faTM_gaussianFitAmplitude':[f['amplitude'] for f in fitted_params_list],
                        'faTomo_gaussianFitX':[f['x0'] for f in tomo_fitted_params_list],
                        'faTomo_gaussianFitY':[f['y0'] for f in tomo_fitted_params_list],
                        'faTomo_gaussianFitZ':[f['z0'] for f in tomo_fitted_params_list],
                        'faTomo_gaussianFitSigmaX':[f['sigma_x'] for f in tomo_fitted_params_list],
                        'faTomo_gaussianFitSigmaY':[f['sigma_y'] for f in tomo_fitted_params_list],
                        'faTomo_gaussianFitSigmaZ':[f['sigma_z'] for f in tomo_fitted_params_list],
                        'faTomo_gaussianFitAmplitude':[f['amplitude'] for f in tomo_fitted_params_list],
                        'faTomo_Surface1':d1,
                        'faTomo_Surface2':d2}
                        )
    aunppositions = folder / "aunps" 
    output_filename = aunppositions / f"aunp_match.star"
    starfile.write(data,output_filename)
    torch.cuda.empty_cache()

async def aunp_match(
    mdoc_done_path: Path,
    destination: Path,
    sample_name: str
):
    from concurrent.futures import ProcessPoolExecutor 

    loop = asyncio.get_event_loop()
    executor = ProcessPoolExecutor(max_workers=1)
    while True:
        with open(mdoc_done_path, "r") as f:
            for mdoc in f.readlines():
                mdoc = mdoc.strip()
                prefix = mdoc.split(".")[0]
                if prefix in state["copy_files"] and prefix not in state["aunp_match"]:
                    aunp_dir = destination / f"{sample_name}_{prefix}" / "aretomo" / "aunps"
                    aunp_dir.mkdir(parents=True, exist_ok=True)
                    if not (aunp_dir / "aunp_match.star").exists():
                        print(f"Running aunp_match for {prefix}")
                        async with gpu_lock:
                            await loop.run_in_executor(executor,run_aunp_match,destination / f"{sample_name}_{prefix}" / "aretomo", f"{sample_name}_{prefix}")
                    state["aunp_match"][prefix] = True
        await asyncio.sleep(1)

async def start_watch(
        mdoc_done_path: Path,
        destination_folder: Path,
        sample_name: str
):
    # Initial state
    multiprocessing.set_start_method('spawn')

    cp_t = copy_files(mdoc_done_path, destination_folder, sample_name)
    ddw_t = ddw(mdoc_done_path, destination_folder, sample_name)
    previews_t = create_previews(mdoc_done_path, destination_folder, sample_name)
    membrainseg_t = membrainseg(mdoc_done_path, destination_folder, sample_name)
    aunp_match_t = aunp_match(mdoc_done_path, destination_folder, sample_name)
    await asyncio.gather(cp_t, ddw_t, previews_t,membrainseg_t,aunp_match_t)

def start_watch_cli(
        mdoc_done_path: Path,
        destination_folder: Path,
        sample_name: str
):
    try:
        asyncio.run(start_watch(mdoc_done_path, destination_folder, sample_name))
    except KeyboardInterrupt:
        print('stopped via KeyboardInterrupt')
    
        

if __name__ == "__main__":
    feud.run(start_watch_cli)