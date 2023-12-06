import typer
from rich.prompt import Prompt
app = typer.Typer()

from findingampa.pipeline1.pipeline1_data import Pipeline1Data, Pipeline1TomogramResults, Pipeline1ClusterRegion
from findingampa.project_management import FAContext, Tomogram
from pathlib import Path
import pandas as pd

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

@app.command()
def run_pipeline1(
    ctx: FAContext,
    skip_subregions: bool = typer.Option(False, help="Skip subregion annotation"),
    skip_threshold: bool = typer.Option(False, help="Skip thresholding"),
    force_contouring: bool = typer.Option(False, help="Force contouring"),
    force_findbeads: bool = typer.Option(False, help="Force findbeads"),
):
    """Run pipeline 1"""

    from pathlib import Path
    from subprocess import Popen, PIPE, TimeoutExpired, run
    import mrcfile
    from time import sleep

    
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
    output_path = ctx.obj.project.Settings.FADataDir / "pipeline_data" / "pipeline1"
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
        segmentation_file = annotate_membrane(ctx, tomogram, output_path)
        
        if tomogram.FATomogramID in data.pl1TomogramResults:
            data.pl1TomogramResults[tomogram.FATomogramID].pl1SegmentationFile = segmentation_file
        else:
            data.pl1TomogramResults[tomogram.FATomogramID] = Pipeline1TomogramResults(pl1TomogramId=tomogram.FATomogramID, pl1SegmentationFile=segmentation_file)
        data.write(data_file)

    # Convert segmentation to contours and split into objects

    for tomogram in ctx.obj.tomograms:
        if tomogram.FATomogramID in data.pl1TomogramResults and data.pl1TomogramResults[tomogram.FATomogramID].pl1SegmentationContourFile is not None and not force_contouring:
            continue
        
        membrain_output_path = output_path / "membrain" / tomogram.FATomogramName
        membrain_output_path.mkdir(parents=True, exist_ok=True)
        # Replace "."
        segment_rotx = str(Path(data.pl1TomogramResults[tomogram.FATomogramID].pl1SegmentationFile).with_suffix("")) + "_rotx.mrc"
        print(segment_rotx)
        membrain_automod_file = membrain_output_path / "membrain_automod.mod"

        run(f"{ctx.obj.project.Settings.FAImodPath}/clip rotx {data.pl1TomogramResults[tomogram.FATomogramID].pl1SegmentationFile} {segment_rotx}", shell=True)
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
        data.pl1TomogramResults[tomogram.FATomogramID].pl1SegmentationContourFile = membrain_surfaces_file_point
        data.write(data_file)


    # Annotate subregions
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
    
                    
    # Run findbeads3d
    for subregion in data.pl1ClusterRegions.values():
        if subregion.pl1AuNPModelfile is not None and not force_findbeads:
            continue
        tomogram = next(tomogram for tomogram in ctx.obj.tomograms if tomogram.FATomogramID == subregion.pl1TomogramId)
        beads_model_file_dir = (output_path / "findbeads")
        beads_model_file_dir.mkdir(parents=True, exist_ok=True)
        beads_model_file = beads_model_file_dir / f"{tomogram.FATomogramName}_{subregion.pl1CluserId}.mod"
        beads_point_file = beads_model_file_dir / f"{tomogram.FATomogramName}_{subregion.pl1CluserId}.txt"
        run(f"{ctx.obj.project.Settings.FAImodPath}/findbeads3d -inp {ctx.obj.project.Settings.FADataDir / tomogram.FABin10Tomogram}  -si 1.5 -a -50,50 -yl -thresh 30 -sto 0.5 -o {beads_model_file} -x {subregion.pl1Xmin},{subregion.pl1Xmax} -z {subregion.pl1Zmin},{subregion.pl1Zmax} -yminmax {subregion.pl1Ymin},{subregion.pl1Ymax}", shell=True)
        run(f"{ctx.obj.project.Settings.FAImodPath}/model2point -ob -v -1 {beads_model_file} {beads_point_file}", shell=True)
        subregion.pl1AuNPModelfile = beads_model_file
        data.write(data_file)

    # Assign threshold and pre- and post-synaptic membranes
    for subregion in data.pl1ClusterRegions.values():
        if skip_threshold:
            continue
        tomogram = next(tomogram for tomogram in ctx.obj.tomograms if tomogram.FATomogramID == subregion.pl1TomogramId)
        
        output_path = ctx.obj.project.Settings.FADataDir / "pipeline_data" / "pipeline1" / "subregions" / f"{tomogram.FATomogramName}_{subregion.pl1CluserId}"
        output_path.mkdir(parents=True, exist_ok=True)\
        
        included_objects = create_model_file_for_subregion(ctx=ctx,
                                                           tomogram=tomogram,
                                                           tomogram_results=data.pl1TomogramResults[tomogram.FATomogramID],
                                                           subregion=subregion,
                                                           output_path=output_path)


        p_3dmod = Popen(f"{ctx.obj.project.Settings.FAImodPath}/3dmod -L {ctx.obj.project.Settings.FADataDir / tomogram.FABin10Tomogram} {output_path / 'temp_model.mod'}",
                shell=True, 
                #cwd=findbeads_path,
                stdin=PIPE, 
                stdout=PIPE, 
                stderr=PIPE,
                bufsize=1,
                universal_newlines=True)
        line = ""
        p_3dmod.stdin.write("8\n")
        while "OK" not in line:
            line = p_3dmod.stderr.readline()
        p_3dmod.stdin.write("3\n")
        line = ""
        while "OK" not in line:
            line = p_3dmod.stderr.readline()
        
        included_objects
        threshold = Prompt.ask("Please enter the desired threshold value", default=str(subregion.pl1AuNPThreshold))
        if subregion.pl1PresynapticObjectId is not None:
            try:
                pre_synaptic = included_objects.get_loc(subregion.pl1PresynapticObjectId) + 2
            except KeyError:
                pre_synaptic = None
        else:
            pre_synaptic = None
        pre_synaptic = int(Prompt.ask("Please enter the model number of the presynaptic membrane", default=str(pre_synaptic)))
        if subregion.pl1PostsynapticObjectId is not None:
            try:
                post_synaptic = included_objects.get_loc(subregion.pl1PostsynapticObjectId) + 2
            except KeyError:
                post_synaptic = None
        else:
            post_synaptic = None
        post_synaptic = int(Prompt.ask("Please enter the model number of the postsynaptic membrane", default=str(post_synaptic)))
            
        subregion.pl1AuNPThreshold = threshold
        subregion.pl1PresynapticObjectId = included_objects[pre_synaptic-2]
        subregion.pl1PostsynapticObjectId = included_objects[post_synaptic-2]
        p_3dmod.terminate()
        
        write_aunp_model_file(ctx=ctx,
                              AuNPModelfile=subregion.pl1AuNPModelfile,
                              threshold=subregion.pl1AuNPThreshold,
                              output_path=output_path / "aunp.txt")
        subregion.pl1AuNPThresholdedModel = output_path / "aunp.txt"
        write_synaptic_contour_model_file(ctx=ctx,
                                          ContourModelfile=data.pl1TomogramResults[tomogram.FATomogramID].pl1SegmentationContourFile,
                                          object_id=subregion.pl1PresynapticObjectId,
                                          output_path=output_path / "presynaptic.txt")
        subregion.pl1PresynapticModel = output_path / "presynaptic.txt"
        write_synaptic_contour_model_file(ctx=ctx,
                                            ContourModelfile=data.pl1TomogramResults[tomogram.FATomogramID].pl1SegmentationContourFile,
                                            object_id=subregion.pl1PostsynapticObjectId,
                                            output_path=output_path / "postsynaptic.txt")
        subregion.pl1PostsynapticModel = output_path / "postsynaptic.txt"
        data.write(data_file)
        

        #run(f"{ctx.obj.project.Settings.FAImodPath}/3dmod -L -m  -x {subregion.pl1Xmin},{subregion.pl1Xmax} -z {subregion.pl1Zmin},{subregion.pl1Zmax} -y {subregion.pl1Ymin},{subregion.pl1Ymax} {ctx.obj.project.Settings.FADataDir / tomogram.FABin10Tomogram} {ctx.obj.project.Settings.FADataDir / 'temp_model.mod'}", shell=True)

    # Define potential AMPA poses


    all_AuNP_pair_distances = []
    all_AuNP_center_to_postsynaptic_distances = []
    all_AuNP_center_to_presynaptic_distances = []
    for subregion in data.pl1ClusterRegions.values():
        from findingampa.find_ampa_poses import find_pair_centers, plot_aunp_distances, find_closest_point_on_membrane, plot_aunp_distance_to_membrane
        tomogram = next(tomogram for tomogram in ctx.obj.tomograms if tomogram.FATomogramID == subregion.pl1TomogramId)
        output_path = ctx.obj.project.Settings.FADataDir / "pipeline_data" / "pipeline1" / "subregions" / f"{tomogram.FATomogramName}_{subregion.pl1CluserId}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        if subregion.pl1AuNPThresholdedModel is not None and subregion.pl1PresynapticModel is not None and subregion.pl1PostsynapticModel is not None:
            import numpy as np
            from scipy.spatial.transform import Rotation as R
            import starfile
            aunp_data = pd.read_csv(subregion.pl1AuNPThresholdedModel, delim_whitespace=True, header=None, names=["o","c","x","y","z","v"])
            postsynaptic_data = pd.read_csv(subregion.pl1PostsynapticModel, delim_whitespace=True, header=None, names=["o","c","x","y","z"])
            presynaptic_data = pd.read_csv(subregion.pl1PresynapticModel, delim_whitespace=True, header=None, names=["o","c","x","y","z"])

            centers, distances = find_pair_centers(aunp_data)
            all_AuNP_pair_distances.extend(distances)
            plot_aunp_distances(distances, output_path / "histogram_pairdistance.png")

            closest_points = find_closest_point_on_membrane(centers, postsynaptic_data)
            closest_points_pre = find_closest_point_on_membrane(centers, presynaptic_data)
            closest_points_post_AuNPs = find_closest_point_on_membrane_to_AuNP(aunp_data, postsynaptic_data)
            np.savetxt('AuNP_distances_to_postsynaptic.txt', closest_points_post_AuNPs)
            closest_points_pre_AuNPs = find_closest_point_on_membrane_to_AuNP(aunp_data, presynaptic_data)
            np.savetxt('AuNP_distances_to_presynaptic.txt', closest_points_pre_AuNPs)

            ampa_positions = []
            results_relion = []
            vector_lengths_post = []
            vector_lengths_pre = []
            for x, (c, cp, cpre) in enumerate(zip(centers, closest_points, closest_points_pre)):
                ampa_positions.append({
                    "o": 1,
                    "c": x+1,
                    "x": c[0] /10,
                    "y": c[1]/10,
                    "z": c[2]/10
                })
                ampa_positions.append({
                    "o": 1,
                    "c": x+1,
                    "x": cp[0]/10,
                    "y": cp[1]/10,
                    "z": cp[2]/10
                })
                vector_pre = c - cpre
                vector_lengths_pre.append(np.linalg.norm(vector_pre))
                vector = c - cp
                vector_lengths_post.append(np.linalg.norm(vector))
                vector_norm = vector / np.linalg.norm(vector)

                # 60 is fudge factor to line out TMP with membrane
                cp = cp + vector_norm * 60
                # Get thre rotation betwen vector and [0,0,1] as a 3D eulerangle

                rot, angle = R.align_vectors([[0, 0, 1]], [vector])
                eulers = rot.as_euler("ZYZ", degrees=True)
                results_relion.append({
                    "rlnCoordinateX": cp[0],
                    "rlnCoordinateY": cp[1],
                    "rlnCoordinateZ": cp[2],
                    "rlnOriginXAngst": 0,
                    "rlnOriginYAngst": 0,
                    "rlnOriginZAngst": 0,
                    "rlnAngleRot": eulers[0],
                    "rlnAngleTilt": eulers[1],
                    "rlnAnglePsi": eulers[2],
                    "rlnImagePixelSize": 1.0,
                    "rlnMicrographName": "micrograph.mrc",
                    "rlnOpticsGroup": 1,
                    "faVectorX": vector_norm[0],
                    "faVectorY": vector_norm[1],
                    "faVectorZ": vector_norm[2],
                })
            all_AuNP_center_to_postsynaptic_distances.extend(vector_lengths_post)
            plot_aunp_distance_to_membrane(vector_lengths_post, output_path / "histogram_distancepost.png", "postsynaptic")
            all_AuNP_center_to_presynaptic_distances.extend(vector_lengths_pre)
            plot_aunp_distance_to_membrane(vector_lengths_pre, output_path / "histogram_distancepre.png", "presynaptic")
            pd.DataFrame(ampa_positions).to_csv(output_path / "ampa_positions.txt", sep=" ", header=None, index=False, mode="w")
            subregion.pl1AMPAPosesModel = output_path / "ampa_positions.txt"
            starfile.write({
                'particles': pd.DataFrame(results_relion), 
                'optics': pd.DataFrame([
                    {'rlnOpticsGroup':1}
                    ])
            }, output_path / "relion_poses.star")
            subregion.pl1AMPAPosesStarfile = output_path / "relion_poses.star"
            data.write(data_file)
    output_path = ctx.obj.project.Settings.FADataDir / "pipeline_data" / "pipeline1" / "subregions"
    plot_aunp_distances(all_AuNP_pair_distances, output_path / "histogram_pairdistance_all.png")
    plot_aunp_distance_to_membrane(all_AuNP_center_to_postsynaptic_distances, output_path / "histogram_distancepost_all.png", "postsynaptic")
    plot_aunp_distance_to_membrane(all_AuNP_center_to_presynaptic_distances, output_path / "histogram_distancepre_all.png", "presynaptic")

@app.command()
def visualize_3dmod(
        ctx: FAContext,
):
    import subprocess
    data = Pipeline1Data.read(ctx.obj.project.Settings.FADataDir / "pipeline_data" / "pipeline1" / "pipeline1_data.star")
    for tomogram in ctx.obj.tomograms:
        subregions = [subregion for subregion in data.pl1ClusterRegions.values() if subregion.pl1TomogramId == tomogram.FATomogramID]
        output_path = ctx.obj.project.Settings.FADataDir / "pipeline_data" / "pipeline1" / "visualization_3dmod" / f"{tomogram.FATomogramName}"
        output_path.mkdir(parents=True, exist_ok=True)
        # Join AuNP models
        aunps = []
        max_c = 0
        for subregion in subregions:
            aunp_data = pd.read_csv(subregion.pl1AuNPThresholdedModel, delim_whitespace=True, header=None, names=["o","c","x","y","z","v"])
            aunp_data["c"] = aunp_data["c"] + max_c
            max_c = max(aunp_data["c"])
            aunps.append(aunp_data)
        combined_aunp = pd.concat(aunps)
        combined_aunp.to_csv(output_path / "combined_aunp.txt", sep=" ", header=None, index=False, mode="w")
        subprocess.run(f"{ctx.obj.project.Settings.FAImodPath}/point2model  -sc -color 255,215,0 -sp 2 {output_path / 'combined_aunp.txt'} {output_path / 'combined_aunp.mod'}", shell=True)

        # Color contour objects
        contour_data  = pd.read_csv(data.pl1TomogramResults[tomogram.FATomogramID].pl1SegmentationContourFile, delim_whitespace=True, header=None, names=["o","c","x","y","z"])
        color_string = ""
        for object in contour_data["o"].unique():
            pre = False
            post = False
            for sr in subregions:
                if sr.pl1PresynapticObjectId == object:
                    pre = True     
                elif sr.pl1PostsynapticObjectId == object:
                    post=True
            if pre:
                color_string += " -color 255,0,0 "
            elif post:
                color_string += " -color 0,0,255 "
            else:   
                color_string += " -color 255,255,255 "
        subprocess.run(f"{ctx.obj.project.Settings.FAImodPath}/point2model  {color_string} {data.pl1TomogramResults[tomogram.FATomogramID].pl1SegmentationContourFile} {output_path / 'colored_contour.mod'}", shell=True)

        # Combined ampa poses
        ampa_poses = []
        max_c = 0
        for subregion in subregions:
            ampa_pose_data = pd.read_csv(subregion.pl1AMPAPosesModel, delim_whitespace=True, header=None, names=["o","c","x","y","z"])
            ampa_pose_data["c"] = ampa_pose_data["c"] + max_c
            max_c = max(ampa_pose_data["c"])
            ampa_poses.append(ampa_pose_data)
        combined_ampa_poses = pd.concat(ampa_poses)
        combined_ampa_poses.to_csv(output_path / "combined_ampa_poses.txt", sep=" ", header=None, index=False, mode="w")
        subprocess.run(f"{ctx.obj.project.Settings.FAImodPath}/point2model  -color 128,0,128 {output_path / 'combined_ampa_poses.txt'} {output_path / 'combined_ampa_poses.mod'}", shell=True)

        # Combine all three models
        subprocess.run(f"{ctx.obj.project.Settings.FAImodPath}/imodjoin {output_path / 'combined_aunp.mod'} {output_path / 'colored_contour.mod'} {output_path / 'combined_ampa_poses.mod'} {output_path / 'combined_all.mod'}", shell=True)
            

# fOR BLENDER:

# iMODMESH CONTOURS
# imod2vrml2 *.wrl file
# Import to blender. Cancel rotations, scale by 0.1
# IMport relion poses
# Still need to figure out what to do about gold