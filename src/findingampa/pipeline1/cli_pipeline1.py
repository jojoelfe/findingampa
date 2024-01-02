import typer
from rich.prompt import Prompt
app = typer.Typer()

from findingampa.pipeline1.pipeline1_data import Pipeline1Data, Pipeline1TomogramResults, Pipeline1ClusterRegion
from findingampa.project_management import FAContext, Tomogram
from pathlib import Path
import pandas as pd







@app.command()
def run_pipeline1(
    ctx: FAContext,
    skip_threshold: bool = typer.Option(False, help="Skip thresholding"),
    force_contouring: bool = typer.Option(False, help="Force contouring"),
    force_findbeads: bool = typer.Option(False, help="Force findbeads"),
):
    """Run pipeline 1"""

    from pathlib import Path
    import mrcfile
    from time import sleep
    from pipeline1_steps import annotate_membrane, convert_qpix_into_imod
    
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
        membrain_surfaces_file_point = convert_qpix_into_imod(ctx, tomogram, output_path, data.pl1TomogramResults[tomogram.FATomogramID].pl1SegmentationFile)

        data.pl1TomogramResults[tomogram.FATomogramID].pl1SegmentationContourFile = membrain_surfaces_file_point
        data.write(data_file)


    # Annotate subregions
    maxclusterid = 0
    if len(data.pl1ClusterRegions) > 0:
        maxclusterid = max(data.pl1ClusterRegions.keys())
    for tomogram in ctx.obj.tomograms:
        if data.pl1TomogramResults[tomogram.FATomogramID]
        annotate_subregion(ctx, tomogram, output_path, data, maxclusterid)
            
    
                    
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
        from findingampa.find_ampa_poses import find_pair_centers, plot_aunp_distances, find_closest_point_on_membrane, find_closest_point_on_membrane_to_AuNP, plot_aunp_distance_to_membrane
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
            
            closest_points_post_AuNPs, postdistances = find_closest_point_on_membrane_to_AuNP(aunp_data, postsynaptic_data)
            np.savetxt(f"{output_path}/{tomogram.FATomogramName}_{subregion.pl1CluserId}_AuNP_distance_to_post.txt", postdistances)
            closest_points_pre_AuNPs, predistances = find_closest_point_on_membrane_to_AuNP(aunp_data, presynaptic_data)
            np.savetxt(f"{output_path}/{tomogram.FATomogramName}_{subregion.pl1CluserId}_AuNP_distance_to_pre.txt", predistances)

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