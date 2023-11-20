import typer
app = typer.Typer()
from membrain_seg.segmentation.segment import segment
from findingampa.pipeline1.pipeline1_data import Pipeline1Data, Pipeline1TomogramResults, Pipeline1ClusterRegion



def update_modelfile(ctx, data, tomogram):
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



@app.command()
def run_pipeline1(
    ctx: typer.Context,
    skip_subregions: bool = typer.Option(False, help="Skip subregion annotation"),
):
    """Run pipeline 1"""

    from pathlib import Path
    from subprocess import Popen, PIPE, TimeoutExpired, run
    import mrcfile
    from time import sleep

    MEMBRAIN_MODEL_PATH =  Path(ctx.obj.project.Settings.FADataDir) / "static" / "MemBrain_seg_v10_alpha.ckpt"
    # Step1: Annotate beads
    # In each tomogram subregions need to be annoated using xminxmax yminymax zminzmax (User rubberbadn in 3dmod)
    # ALso each subregion need an optimized threshold value
    # findbeads3d command is: 
    # findbeads3d -inp $bin10tomo  -c c.mod -si 1.5 -a -50,50 -yl -thresh 30 -sto $threshold -o aunp_$i.mod -x $xminxmax -z zminzmax -yminmax yminymax

    # Step2: Annotate membranes
    # Command is:
    # membrain segment --tomogram-path $binnedtomo --ckpt-path static/MemBrain_seg_v10_alpha.ckpt

    # Step3: Sort and split surfaces
    # Command is:
    # imodauto -h 1.1 #membrainsegmentation  membrane_contours.mod
    # imodsortsurf -s  membrane_contours.mod membrane_surfaces.mod
    # Identify postsynaptic membranes in 3dmod and write down numbers
    # Extract using:
    # imodextract $number membrane_surfaces.mod postsynatpic_membrane_$i.mod
    
    # Step4: Estimate AMPA poses
    # Needs as input a dictionary describing which AuNP patch is on which postsynaptic membrane
    #files_to_analyze = [
    #    [data_folder / "aunp1.mod", data_folder / "postsynaptic2.mod"],
    #    [data_folder / "aunp2.mod", data_folder / "postsynaptic3.mod"],
    #    [data_folder / "aunp3.mod", data_folder / "postsynaptic1.mod"],
    #    [data_folder / "aunp4.mod", data_folder / "postsynaptic1.mod"],
    #]
    # run code currently in find_ampa_poses.py

    # Two interactive steps: 

    # Assign subareas
    # Iterate over subareas, adjust threshold and confirm or set post/pre assignment
    output_path = Path(ctx.obj.project.Settings.FADataDir) / "pipeline_data" / "pipeline1"
    output_path.mkdir(parents=True, exist_ok=True)
    data_file = output_path / "pipeline1_data.star"
    if data_file.exists():
        data = Pipeline1Data.read(data_file)
    else:
        data = Pipeline1Data()
        data.write(data_file)
    typer.echo(data)
    
    # Annotate membranes

    for tomogram in ctx.obj.tomograms:
        # Skip if tomogram already segmented
        if tomogram.FATomogramID in data.pl1TomogramResults and data.pl1TomogramResults[tomogram.FATomogramID].pl1SegmentationFile is not None:
            continue

        membrain_output_path = output_path / "membrain" / tomogram.FATomogramName
        membrain_output_path.mkdir(parents=True, exist_ok=True)

        segmentation_file = segment(
            tomogram_path=ctx.obj.project.Settings.FADataDir / tomogram.FABin10Tomogram,
            ckpt_path=MEMBRAIN_MODEL_PATH,
            out_folder=membrain_output_path,
        )
        if tomogram.FATomogramID in data.pl1TomogramResults:
            data.pl1TomogramResults[tomogram.FATomogramID].pl1SegmentationFile = segmentation_file
        else:
            data.pl1TomogramResults[tomogram.FATomogramID] = Pipeline1TomogramResults(pl1TomogramId=tomogram.FATomogramID, pl1SegmentationFile=segmentation_file)
        data.write(data_file)

    # Convert segmentation to contours and split into objects

    for tomogram in ctx.obj.tomograms:
        if tomogram.FATomogramID in data.pl1TomogramResults and data.pl1TomogramResults[tomogram.FATomogramID].pl1SegmentationContourFile is not None:
            continue
        
        membrain_output_path = output_path / "membrain" / tomogram.FATomogramName
        membrain_output_path.mkdir(parents=True, exist_ok=True)

        membrain_automod_file = membrain_output_path / "membrain_automod.mod"

        run(f"{ctx.obj.project.Settings.FAImodPath}/imodauto -h 1.1 {data.pl1TomogramResults[tomogram.FATomogramID].pl1SegmentationFile} {membrain_automod_file}", shell=True)
        run(f"{ctx.obj.project.Settings.FAImodPath}/imodmesh {membrain_automod_file}", shell=True)
        membrain_surfaces_file = membrain_output_path / "membrain_surfaces.mod"
        run(f"{ctx.obj.project.Settings.FAImodPath}/imodsortsurf -s {membrain_automod_file} {membrain_surfaces_file}", shell=True)
        data.pl1TomogramResults[tomogram.FATomogramID].pl1SegmentationContourFile = membrain_surfaces_file
        data.write(data_file)





    # Annotate subaregions
    maxclusterid = 0
    if len(data.pl1ClusterRegions) > 0:
        maxclusterid = max(data.pl1ClusterRegions.keys())
    if not skip_subregions:
        for tomogram in ctx.obj.tomograms:
            
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
    
                    
    # Run findbeads3d
    for subregion in data.pl1ClusterRegions.values():
        if subregion.pl1AuNPModelfile is not None:
            continue
        tomogram = next(tomogram for tomogram in ctx.obj.tomograms if tomogram.FATomogramID == subregion.pl1TomogramId)
        beads_model_file = (output_path / "findbeads")
        beads_model_file.mkdir(parents=True, exist_ok=True)
        beads_model_file = beads_model_file/ f"{subregion.pl1CluserId}.mod"
        run(f"{ctx.obj.project.Settings.FAImodPath}/findbeads3d -inp {ctx.obj.project.Settings.FADataDir / tomogram.FABin10Tomogram}  -si 1.5 -a -50,50 -yl -thresh 30 -sto 0.5 -o {beads_model_file} -x {subregion.pl1Xmin},{subregion.pl1Xmax} -z {subregion.pl1Zmin},{subregion.pl1Zmax} -yminmax {subregion.pl1Ymin},{subregion.pl1Ymax}", shell=True)
        subregion.pl1AuNPModelfile = beads_model_file
        data.write(data_file)

    # Assign threshold and pre- and post-synaptic membranes
    for subregion in data.pl1ClusterRegions.values():
        tomogram = next(tomogram for tomogram in ctx.obj.tomograms if tomogram.FATomogramID == subregion.pl1TomogramId)
        typer.echo(f"{ctx.obj.project.Settings.FAImodPath}/3dmod -L  -x {subregion.pl1Xmin},{subregion.pl1Xmax} -z {subregion.pl1Zmin},{subregion.pl1Zmax} -y {subregion.pl1Ymin},{subregion.pl1Ymax} {ctx.obj.project.Settings.FADataDir / tomogram.FABin10Tomogram} {data.pl1TomogramResults[subregion.pl1TomogramId].pl1SegmentationContourFile}")
        #run(f"{ctx.obj.project.Settings.FAImodPath}/3dmod -L -m  -x {subregion.pl1Xmin},{subregion.pl1Xmax} -z {subregion.pl1Zmin},{subregion.pl1Zmax} -y {subregion.pl1Ymin},{subregion.pl1Ymax} {ctx.obj.project.Settings.FADataDir / tomogram.FABin10Tomogram} {data.pl1TomogramResults[subregion.pl1TomogramId].pl1SegmentationContourFile}", shell=True)
        #
