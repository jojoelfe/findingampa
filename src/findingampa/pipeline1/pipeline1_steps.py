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
    return Path(segmentation_file)

def convert_qpix_into_imod(ctx, tomogram, output_path, segmentation_file):
    from subprocess import run

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



def create_model_file_for_subregion(ctx: FAContext, 
                                    tomogram: Tomogram, 
                                    tomogram_results: Pipeline1TomogramResults,
                                    subregion: Pipeline1ClusterRegion, 
                                    output_path: Path):
    import pandas as pd
    import colorcet as cc
    import numpy as np

    #aunp_data = pd.read_csv(aunp_file, delim_whitespace=True, header=None, names=["ob","c","x","y","z","v"])
    contour_data = pd.read_csv(ctx.obj.project.Settings.FADataDir / tomogram_results.pl1SegmentationContourFile, delim_whitespace=True, header=None, names=["ob","c","x","y","z"])

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
    print()
    # Convert to model file
    import subprocess
    subprocess.run(f"{ctx.obj.project.Settings.FAImodPath}/point2model {output_path / 'temp_model.txt'} {output_path / 'temp_model.mod'}", shell=True)
    subprocess.run(f"{ctx.obj.project.Settings.FAImodPath}/imodjoin -c {ctx.obj.project.Settings.FADataDir / subregion.pl1AuNPModelfile}  {output_path/ 'temp_model.mod'} {output_path/ 'temp_model.mod'}", shell=True)
    
    return unique_objects 

def run_findbeads3d(ctx: FAContext, subregion: Pipeline1ClusterRegion, output_path: Path):
    from subprocess import run

    tomogram = next(tomogram for tomogram in ctx.obj.tomograms if tomogram.FATomogramID == subregion.pl1TomogramId)
    beads_model_file_dir = (output_path / "findbeads")
    beads_model_file_dir.mkdir(parents=True, exist_ok=True)
    beads_model_file = beads_model_file_dir / f"{tomogram.FATomogramName}_{subregion.pl1CluserId}.mod"
    beads_point_file = beads_model_file_dir / f"{tomogram.FATomogramName}_{subregion.pl1CluserId}.txt"
    run(f"{ctx.obj.project.Settings.FAImodPath}/findbeads3d -inp {ctx.obj.project.Settings.FADataDir / tomogram.FABin10Tomogram}  -si 1.5 -a -50,50 -yl -thresh 30 -sto 0.5 -o {beads_model_file} -x {subregion.pl1Xmin},{subregion.pl1Xmax} -z {subregion.pl1Zmin},{subregion.pl1Zmax} -yminmax {subregion.pl1Ymin},{subregion.pl1Ymax}", shell=True)
    run(f"{ctx.obj.project.Settings.FAImodPath}/model2point -ob -v -1 {beads_model_file} {beads_point_file}", shell=True)
    subregion.pl1AuNPModelfile = beads_model_file.relative_to(ctx.obj.project.Settings.FADataDir)

def write_aunp_model_file(
        ctx: FAContext,
        AuNPModelfile: Path,
        threshold: float,
        output_path: Path,
):
    import subprocess
    import pandas as pd
    subprocess.run(f"{ctx.obj.project.Settings.FAImodPath}/model2point -ob -v -1 {ctx.obj.project.Settings.FADataDir / AuNPModelfile} {output_path}" , shell=True)
    aunp_data = pd.read_csv(output_path, delim_whitespace=True, header=None, names=["ob","c","x","y","z","v"])
    aunp_data = aunp_data[aunp_data["v"] >= threshold]
    aunp_data.to_csv(output_path, sep=" ", header=None, index=False, mode="w")

def write_synaptic_contour_model_file(
        ctx: FAContext,
        ContourModelfile: Path,
        object_id: int,
        output_path: Path,
):
    import pandas as pd

    contour_data = pd.read_csv(ContourModelfile, delim_whitespace=True, header=None, names=["ob","c","x","y","z"])
    contour_data = contour_data[contour_data["ob"] == object_id]
    contour_data.to_csv(output_path, sep=" ", header=None, index=False, mode="w")

def estimate_ampa_poses(
        ctx: FAContext,
        subregion: Pipeline1ClusterRegion,
):
    from findingampa.find_ampa_poses import find_pair_centers, plot_aunp_distances, find_closest_point_on_membrane, find_closest_point_on_membrane_to_AuNP, plot_aunp_distance_to_membrane
    tomogram = next(tomogram for tomogram in ctx.obj.tomograms if tomogram.FATomogramID == subregion.pl1TomogramId)
    output_path = ctx.obj.project.Settings.FADataDir / "pipeline_data" / "pipeline1" / "subregions" / f"{tomogram.FATomogramName}_{subregion.pl1CluserId}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    import starfile
    import pandas as pd
    aunp_data = pd.read_csv(ctx.obj.project.Settings.FADataDir / subregion.pl1AuNPThresholdedModel, delim_whitespace=True, header=None, names=["o","c","x","y","z","v"])
    postsynaptic_data = pd.read_csv(ctx.obj.project.Settings.FADataDir / subregion.pl1PostsynapticModel, delim_whitespace=True, header=None, names=["o","c","x","y","z"])
    presynaptic_data = pd.read_csv(ctx.obj.project.Settings.FADataDir / subregion.pl1PresynapticModel, delim_whitespace=True, header=None, names=["o","c","x","y","z"])

    pairs = find_pair_centers(aunp_data)
    if len(pairs) == 0:
        return
    print(pairs)
    center_coordinates = pairs[:,2:]

    closest_points = find_closest_point_on_membrane(center_coordinates, postsynaptic_data)
    closest_points_pre = find_closest_point_on_membrane(center_coordinates, presynaptic_data)
    
    
    ampa_positions = []
    results_relion = []

    aunp_coordinates = aunp_data[["x", "y", "z"]].to_numpy() * 10
    for x, (pair, cp, cpre) in enumerate(zip(pairs, closest_points, closest_points_pre)):
        ampa_positions.append({
            "o": 1,
            "c": x+1,
            "x": aunp_coordinates[int(pair[0])][0]/10,
            "y": aunp_coordinates[int(pair[0])][1]/10,
            "z": aunp_coordinates[int(pair[0])][2]/10
        })
        ampa_positions.append({
            "o": 1,
            "c": x+1,
            "x": aunp_coordinates[int(pair[1])][0]/10,
            "y": aunp_coordinates[int(pair[1])][1]/10,
            "z": aunp_coordinates[int(pair[1])][2]/10
        })
        ampa_positions.append({
            "o": 1,
            "c": x+1,
            "x": cp[0]/10,
            "y": cp[1]/10,
            "z": cp[2]/10
        })

        vector_center = pair[2:] - cp
        vector_aunp1 = aunp_coordinates[int(pair[0])] - cp
        vector_aunp2 = aunp_coordinates[int(pair[1])] - cp
        vector_norm = vector_center / np.linalg.norm(vector_center)

        # 60 is fudge factor to line out TMP with membrane
        cp = cp + vector_norm * 60
        # Get thre rotation betwen vector and [0,0,1] as a 3D eulerangle

        rot, _ = R.align_vectors([[-0.5, 0, 1],[0.5,0,1]], [vector_aunp1,vector_aunp2])
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
    
    pd.DataFrame(ampa_positions).to_csv(output_path / "ampa_positions.txt", sep=" ", header=None, index=False, mode="w")
    subregion.pl1AMPAPosesModel = output_path / "ampa_positions.txt"
    starfile.write({
        'particles': pd.DataFrame(results_relion), 
        'optics': pd.DataFrame([
            {'rlnOpticsGroup':1}
            ])
    }, output_path / "relion_poses.star")
    subregion.pl1AMPAPosesStarfile = output_path / "relion_poses.star"

def create_visualization_files(
    ctx: FAContext,
    tomogram: Tomogram,
    data: Pipeline1Data,
):
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
            

from pydantic import BaseModel
class AuNP_statistic(BaseModel):
    distance_to_pre: float
    distance_to_post: float

class AuNP_pair_statistic(BaseModel):
    distance: float
    center_distance_to_pre: float
    center_distance_to_post: float
    chosen: bool = False

def calculate_statistics(ctx: FAContext, subregion: Pipeline1ClusterRegion):
    closest_points_post_AuNPs, postdistances = find_closest_point_on_membrane_to_AuNP(aunp_data, postsynaptic_data)
    np.savetxt(f"{output_path}/{tomogram.FATomogramName}_{subregion.pl1CluserId}_AuNP_distance_to_post.txt", postdistances)
    closest_points_pre_AuNPs, predistances = find_closest_point_on_membrane_to_AuNP(aunp_data, presynaptic_data)
    np.savetxt(f"{output_path}/{tomogram.FATomogramName}_{subregion.pl1CluserId}_AuNP_distance_to_pre.txt", predistances)

    pass

class relion_particle(BaseModel):
    rlnCoordinateX: float
    rlnCoordinateY: float
    rlnCoordinateZ: float
    rlnOriginXAngst: float
    rlnOriginYAngst: float
    rlnOriginZAngst: float
    rlnAngleRot: float
    rlnAngleTilt: float
    rlnAnglePsi: float
    rlnImagePixelSize: float
    rlnMicrographName: str
    rlnOpticsGroup: int
    faVectorX: float
    faVectorY: float
    faVectorZ: float

class relion_optics_group(BaseModel):
    rlnOpticsGroup: int
    