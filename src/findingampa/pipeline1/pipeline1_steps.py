from findingampa.project_management import FAContext, Tomogram
from findingampa.pipeline1.pipeline1_data import Pipeline1Data, Pipeline1TomogramResults, Pipeline1ClusterRegion
from pathlib import Path


def annotate_membrane(ctx, tomogram, output_path):
    from membrain_seg.segmentation.segment import segment

    membrain_output_path = output_path / "membrain" / tomogram.FATomogramName
    membrain_output_path.mkdir(parents=True, exist_ok=True)
    MEMBRAIN_MODEL_PATH =  ctx.obj.project.Settings.FADataDir / "static" / "MemBrain_seg_v10_alpha.ckpt"
    segmentation_file = segment(
        tomogram_path=ctx.obj.project.Settings.FADataDir / tomogram.FABin10Tomogram,
        ckpt_path=MEMBRAIN_MODEL_PATH,
        out_folder=membrain_output_path,
    )
    return segmentation_file

def convert_qpix_into_imod(ctx, tomogram, output_path, segmentation_file):
    from subprocess import Popen, PIPE, TimeoutExpired, run

    membrain_output_path = output_path / "membrain" / tomogram.FATomogramName
    membrain_output_path.mkdir(parents=True, exist_ok=True)
    # Replace "."
    segment_rotx = str(Path(segmentation_file).with_suffix("")) + "_rotx.mrc"
    print(segment_rotx)
    membrain_automod_file = membrain_output_path / "membrain_automod.mod"

    run(f"{ctx.obj.project.Settings.FAImodPath}/clip rotx {segmentation_file} {segment_rotx}", shell=True)
    run(f"{ctx.obj.project.Settings.FAImodPath}/imodauto -n -E 1.0 -u {segment_rotx} {membrain_automod_file}", shell=True)
    run(f"{ctx.obj.project.Settings.FAImodPath}/reducecont -sp 1.0 {membrain_automod_file} {membrain_automod_file}", shell=True)
    run(f"{ctx.obj.project.Settings.FAImodPath}/imodmesh {membrain_automod_file}", shell=True)
    membrain_surfaces_file_rotx = membrain_output_path / "membrain_surfaces_rotx.mod"
    membrain_surfaces_file = membrain_output_path / "membrain_surfaces.mod"
    membrain_surfaces_file_point = membrain_output_path / "membrain_surfaces.txt"
    run(f"{ctx.obj.project.Settings.FAImodPath}/imodsortsurf -s {membrain_automod_file} {membrain_surfaces_file_rotx}", shell=True)
    run(f"{ctx.obj.project.Settings.FAImodPath}/imodtrans -R 0 {membrain_surfaces_file_rotx} {membrain_surfaces_file}", shell=True)
    run(f"{ctx.obj.project.Settings.FAImodPath}/model2point -ob {membrain_surfaces_file} {membrain_surfaces_file_point}", shell=True)
    run(f"{ctx.obj.project.Settings.FAImodPath}/point2model {membrain_surfaces_file_point} {membrain_surfaces_file}", shell=True)

    return membrain_surfaces_file_point

def annotate_subregion(ctx, tomogram, output_path, subregion):
    with mrcfile.open(ctx.obj.project.Settings.FADataDir / tomogram.FABin10Tomogram) as mrc:
                ydim = mrc.header.ny
                ydim = mrc.header.ny
                xdim = mrc.header.nx
    #findbeads_path = output_path / "findbeads" / tomogram.FATomogramName
    #findbeads_path.mkdir(parents=True, exist_ok=True)

    # Launch 3dmod in a way that allows read/write to stdin/stdout

    p_3dmod  = Popen(f"{ctx.obj.project.Settings.FAImodPath}/3dmod -Y -L {ctx.obj.project.Settings.FADataDir / tomogram.FABin10Tomogram}", 
                    shell=True, 
                    #cwd=findbeads_path,
                    stdin=PIPE, 
                    stdout=PIPE, 
                    stderr=PIPE,
                    bufsize=1,
                    universal_newlines=True)
    sleep(1)
    while p_3dmod.poll() is None:
        sleep(0.1)
        modelfilename = update_modelfile(ctx, data, tomogram)
        print(modelfilename)
        p_3dmod.stdin.write(f"7 {modelfilename}\n")
        sleep(0.1)
        readnextline = True
        while readnextline and p_3dmod.poll() is None:
            line = p_3dmod.stderr.readline()
            print(f"Up {line}")
            if "OK" in line:
                readnextline = False
        # Wait for user to confirm
        typer.confirm("Please annotate subregions in 3dmod or close 3dmod and press enter to continue",abort=False)
        # Send commands to 3dmod
        if p_3dmod.poll() is not None:
            break
        p_3dmod.stdin.write("10\n")
        sleep(0.1)
        readnextline = True
        found_rubberband = False
        
        while readnextline and p_3dmod.poll() is None:
            line = p_3dmod.stderr.readline()
            if "OK" in line:
                line = p_3dmod.stderr.readline()
            print("Down ", line)
            if "Rubberband" in line:
                found_rubberband = True
                print(f"Found {line}")
                values = line.split()
                if len(values) < 7:
                    typer.echo("Please also assign High and Low Z values")
                    break
                # Read values, taking roation around x into account
                _ , xmin, zmin, xmax, zmax, ymin, ymax = values
                #typer.echo(ctx, output_path, tomogram, xmin, ymin, xmax, ymax, zmin, zmax)
                ymin = ydim - int(ymin)
                ymax = ydim - int(ymax)
                ymin, ymax = ymax, ymin
                print(f"Adding subregion Xmin {xmin} Xmax {xmax} Ymin {ymin} Ymax {ymax} Zmin {zmin} Zmax {zmax}")
                data.pl1ClusterRegions[maxclusterid+1] = Pipeline1ClusterRegion(
                    pl1CluserId=maxclusterid+1,
                    pl1TomogramId=tomogram.FATomogramID,
                    pl1Xmin=xmin,
                    pl1Xmax=xmax,
                    pl1Ymin=ymin,
                    pl1Ymax=ymax,
                    pl1Zmin=zmin,
                    pl1Zmax=zmax,
                )
                maxclusterid += 1
                data.write(data_file)
            if "OK" in line:
                readnextline = False
                if not found_rubberband:
                    typer.echo("Please select a region using rubberband")
                    break

def update_modelfile(ctx: FAContext, data, tomogram):
    import pandas as pd
    import itertools
    import subprocess
    from pathlib import Path
    import tempfile
    # Generate temporary model file
    temp_dir = tempfile.mkdtemp()
    model_path = Path(temp_dir) / "c.mod"
    txt_path = Path(temp_dir) / "c.txt"
    
    model_items = []
    object_id = 0
    for subregion in data.pl1ClusterRegions.values():
        if subregion.pl1TomogramId == tomogram.FATomogramID:
            object_id += 1
            
            for y in range(subregion.pl1Ymin,subregion.pl1Ymax+1):
                model_items.append([object_id, y, subregion.pl1Xmin, y, subregion.pl1Zmin])
                model_items.append([object_id, y, subregion.pl1Xmax, y, subregion.pl1Zmin])
                model_items.append([object_id, y, subregion.pl1Xmax, y, subregion.pl1Zmax])
                model_items.append([object_id, y, subregion.pl1Xmin, y, subregion.pl1Zmax])
                
    
    df = pd.DataFrame(model_items, columns=["obj","con","x","y","z"])
    df.to_csv(txt_path, sep=" ", header=None, index=False, mode="a")
    subprocess.run(f"{ctx.obj.project.Settings.FAImodPath}/point2model {txt_path} {model_path}", shell=True)
    return model_path

def create_model_file_for_subregion(ctx: FAContext, 
                                    tomogram: Tomogram, 
                                    tomogram_results: Pipeline1TomogramResults,
                                    subregion: Pipeline1ClusterRegion, 
                                    output_path: Path) -> pd.Index:
    import pandas as pd
    import colorcet as cc
    import numpy as np

    #aunp_data = pd.read_csv(aunp_file, delim_whitespace=True, header=None, names=["ob","c","x","y","z","v"])
    contour_data = pd.read_csv(tomogram_results.pl1SegmentationContourFile, delim_whitespace=True, header=None, names=["ob","c","x","y","z"])

    #contour_data["ob"] = contour_data["ob"] + 1
    contour_data["v"] = 0
    # Crop aunp data accordind to subregion
    #aunp_data = aunp_data[
    #    (aunp_data["x"] >= subregion.pl1Xmin) &
    #    (aunp_data["x"] <= subregion.pl1Xmax) &
    #    (aunp_data["y"] >= subregion.pl1Ymin) &
    #    (aunp_data["y"] <= subregion.pl1Ymax) &
    #    (aunp_data["z"] >= subregion.pl1Zmin) &
    #    (aunp_data["z"] <= subregion.pl1Zmax)
    #]
    # Crop contour data according to subregion
    contour_data = contour_data[
        (contour_data["x"] >= subregion.pl1Xmin) &
        (contour_data["x"] <= subregion.pl1Xmax) &
        (contour_data["y"] >= subregion.pl1Ymin) &
        (contour_data["y"] <= subregion.pl1Ymax) &
        (contour_data["z"] >= subregion.pl1Zmin) &
        (contour_data["z"] <= subregion.pl1Zmax)
    ]
    contour_data["ob"], unique_objects = pd.factorize(contour_data["ob"])
    contour_data["ob"] = contour_data["ob"] + 1
    print(unique_objects) 

    # Concatenate contour_data to aunpdata
    temp_model_data = pd.concat([contour_data]) # , contour_data])
    # Write to file
    temp_model_data.to_csv(output_path / "temp_model.txt", sep=" ", header=None, index=False, mode="w")

    # Convert to model file
    import subprocess
    subprocess.run(f"{ctx.obj.project.Settings.FAImodPath}/point2model {output_path / 'temp_model.txt'} {output_path / 'temp_model.mod'}", shell=True)
    subprocess.run(f"{ctx.obj.project.Settings.FAImodPath}/imodjoin -c {subregion.pl1AuNPModelfile}  {output_path/ 'temp_model.mod'} {output_path/ 'temp_model.mod'}", shell=True)
    
    return unique_objects 

def write_aunp_model_file(
        ctx: FAContext,
        AuNPModelfile: Path,
        threshold: float,
        output_path: Path,
):
    import subprocess
    subprocess.run(f"{ctx.obj.project.Settings.FAImodPath}/model2point -ob -v -1 {AuNPModelfile} {output_path}" , shell=True)
    aunp_data = pd.read_csv(output_path, delim_whitespace=True, header=None, names=["ob","c","x","y","z","v"])
    print(aunp_data)
    aunp_data = aunp_data[aunp_data["v"] >= threshold]
    aunp_data.to_csv(output_path, sep=" ", header=None, index=False, mode="w")

def write_synaptic_contour_model_file(
        ctx: FAContext,
        ContourModelfile: Path,
        object_id: int,
        output_path: Path,
):
    contour_data = pd.read_csv(ContourModelfile, delim_whitespace=True, header=None, names=["ob","c","x","y","z"])
    contour_data = contour_data[contour_data["ob"] == object_id]
    contour_data.to_csv(output_path, sep=" ", header=None, index=False, mode="w")