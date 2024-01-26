from findingampa.project_management import FAContext, Tomogram
from findingampa.pipeline1.pipeline1_data import Pipeline1Data, Pipeline1TomogramResults, Pipeline1ClusterRegion
from pathlib import Path
from pydantic import BaseModel


def annotate_membrane(ctx: FAContext, tomogram:Tomogram, output_path):
    from membrain_seg.segmentation.segment import segment

    membrain_output_path = output_path / "membrain" / tomogram.FATomogramName
    membrain_output_path.mkdir(parents=True, exist_ok=True)
    MEMBRAIN_MODEL_PATH =  ctx.obj.project_directory / "static" / "MemBrain_seg_v10_alpha.ckpt"
    segmentation_file = segment(
        tomogram_path=ctx.obj.project_directory.parent / (ctx.obj.relion_tomograms[tomogram.FATomogramName].rlnTomoTiltSeriesName[:-4] + tomogram.FAPL1TomoSuffixForMembrain),
        ckpt_path=MEMBRAIN_MODEL_PATH,
        out_folder=membrain_output_path,
    )
    return Path(segmentation_file)

def convert_qpix_into_imod(ctx, tomogram: Tomogram, output_path, segmentation_file):
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
    print()
    # Convert to model file
    import subprocess
    subprocess.run(f"{ctx.obj.project.Settings.FAImodPath}/point2model {output_path / 'temp_model.txt'} {output_path / 'temp_model.mod'}", shell=True)
    subprocess.run(f"{ctx.obj.project.Settings.FAImodPath}/imodjoin -c {subregion.pl1AuNPModelfile}  {output_path/ 'temp_model.mod'} {output_path/ 'temp_model.mod'}", shell=True)
    
    return unique_objects 

def run_findbeads3d(ctx: FAContext, subregion: Pipeline1ClusterRegion, output_path: Path):
    from subprocess import run

    tomogram = next(tomogram for tomogram in ctx.obj.tomograms if tomogram.FATomogramName == subregion.pl1TomogramName)
    beads_model_file_dir = (output_path / "findbeads")
    beads_model_file_dir.mkdir(parents=True, exist_ok=True)
    beads_model_file = beads_model_file_dir / f"{tomogram.FATomogramName}_{subregion.pl1CluserId}.mod"
    beads_point_file = beads_model_file_dir / f"{tomogram.FATomogramName}_{subregion.pl1CluserId}.txt"
    tomogram_path = ctx.obj.project_directory.parent / (ctx.obj.relion_tomograms[tomogram.FATomogramName].rlnTomoTiltSeriesName[:-4] + tomogram.FAPL1TomoSuffixForFindbeads)
    run(f"{ctx.obj.project.Settings.FAImodPath}/findbeads3d -inp {tomogram_path}  -si 1.5 -a -50,50 -yl -thresh 20 -sto 0.5 -o {beads_model_file} -x {subregion.pl1Xmin},{subregion.pl1Xmax} -z {subregion.pl1Zmin},{subregion.pl1Zmax} -yminmax {subregion.pl1Ymin},{subregion.pl1Ymax}", shell=True)
    run(f"{ctx.obj.project.Settings.FAImodPath}/model2point -ob -v -1 {beads_model_file} {beads_point_file}", shell=True)
    subregion.pl1AuNPModelfile= beads_model_file

def write_aunp_model_file(
        ctx: FAContext,
        AuNPModelfile: Path,
        threshold: float,
        output_path: Path,
):
    import subprocess
    import pandas as pd
    subprocess.run(f"{ctx.obj.project.Settings.FAImodPath}/model2point -ob -v -1 {AuNPModelfile} {output_path}" , shell=True)
    aunp_data = pd.read_csv(output_path, delim_whitespace=True, header=None, names=["ob","c","x","y","z","v"])
    original_length = len(aunp_data)
    aunp_data = aunp_data[aunp_data["v"] >= threshold]
    from rich import print
    print(f"Threshold {threshold} selected {len(aunp_data)} out of {original_length} initial picks")
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

class relion_particle(BaseModel):
    rlnTomoName: str
    rlnCoordinateX: float
    rlnCoordinateY: float
    rlnCoordinateZ: float
    rlnAngleRot: float
    rlnAngleTilt: float
    rlnAnglePsi: float

def estimate_ampa_poses(
        ctx: FAContext,
        subregion: Pipeline1ClusterRegion,
):
    from findingampa.find_ampa_poses import find_pair_centers, plot_aunp_distances, find_closest_point_on_membrane, find_closest_point_on_membrane_to_AuNP, plot_aunp_distance_to_membrane
    import numpy as np

    tomogram = next(tomogram for tomogram in ctx.obj.tomograms if tomogram.FATomogramName == subregion.pl1TomogramName)
    output_path = Path("finding_ampa") / "pipeline_data" / "pipeline1" / "subregions" / f"{tomogram.FATomogramName}_{subregion.pl1CluserId}"
    output_path.mkdir(parents=True, exist_ok=True)
    import mrcfile
    tomogram_path = ctx.obj.project_directory.parent / (ctx.obj.relion_tomograms[tomogram.FATomogramName].rlnTomoTiltSeriesName[:-4] + tomogram.FAPL1TomoSuffixForFindbeads)
    with mrcfile.open(tomogram_path) as mrc:
                ydim = mrc.header.ny
                xdim = mrc.header.nx
                zdim = mrc.header.nz
    bin_factor_z = ctx.obj.relion_tomograms[tomogram.FATomogramName].rlnTomoSizeZ / zdim
    bin_factor_y = ctx.obj.relion_tomograms[tomogram.FATomogramName].rlnTomoSizeY / ydim
    bin_factor_x = ctx.obj.relion_tomograms[tomogram.FATomogramName].rlnTomoSizeX / xdim
    binning_factor = np.mean([bin_factor_x, bin_factor_y, bin_factor_z])
    print(binning_factor)
    from scipy.spatial.transform import Rotation as R
    import starfile
    import pandas as pd
    aunp_data = pd.read_csv(subregion.pl1AuNPThresholdedModel, delim_whitespace=True, header=None, names=["o","c","x","y","z","v"])
    postsynaptic_data = pd.read_csv(subregion.pl1PostsynapticModel, delim_whitespace=True, header=None, names=["o","c","x","y","z"])
    presynaptic_data = pd.read_csv(subregion.pl1PresynapticModel, delim_whitespace=True, header=None, names=["o","c","x","y","z"])

    pairs = find_pair_centers(aunp_data)
    if len(pairs) == 0:
        return
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

        # Get the binning factor
        
        results_relion.append({
            "rlnTomoName": tomogram.FATomogramName,
            "rlnCoordinateX": (cp[0]/10) * binning_factor,
            "rlnCoordinateY": (cp[1]/10) * binning_factor,
            "rlnCoordinateZ": (cp[2]/10) * binning_factor,

            "rlnAngleRot": eulers[0],
            "rlnAngleTilt": eulers[1],
            "rlnAnglePsi": eulers[2],
            # "rlnImagePixelSize": ctx.obj.relion_tomograms[tomogram.FATomogramName].rlnTomoTiltSeriesPixelSize,
            # "rlnMicrographName": "micrograph.mrc",
            # "rlnOpticsGroup": 1,
            # "faVectorX": vector_norm[0],
            # "faVectorY": vector_norm[1],
            # "faVectorZ": vector_norm[2],
        })
    
    pd.DataFrame(ampa_positions).to_csv(output_path / "ampa_positions.txt", sep=" ", header=None, index=False, mode="w")
    subregion.pl1AMPAPosesModel = output_path / "ampa_positions.txt"
    starfile.write({
        'particles': pd.DataFrame(results_relion), 
        'optics': pd.DataFrame([
            {'rlnOpticsGroup':1}
            ])
    }, output_path / "relion_poses.star")

def create_visualization_files(
    ctx: FAContext,
    tomogram: Tomogram,
    data: Pipeline1Data,
):
    from subprocess import run
    import pandas as pd
    subregions = [subregion for subregion in data.pl1ClusterRegions.values() if subregion.pl1TomogramName == tomogram.FATomogramName]
    output_path = Path("finding_ampa") / "pipeline_data" / "pipeline1" / "visualization" / f"{tomogram.FATomogramName}"
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
    run(f"{ctx.obj.project.Settings.FAImodPath}/point2model  -sc -color 255,215,0 -sp 2 {output_path / 'combined_aunp.txt'} {output_path / 'combined_aunp.mod'}", shell=True)

    # Color contour objects
    contour_data  = pd.read_csv(data.pl1TomogramResults[tomogram.FATomogramName].pl1SegmentationContourFile, delim_whitespace=True, header=None, names=["o","c","x","y","z"])
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
    run(f"{ctx.obj.project.Settings.FAImodPath}/point2model  {color_string} {data.pl1TomogramResults[tomogram.FATomogramName].pl1SegmentationContourFile} {output_path / 'colored_contour.mod'}", shell=True)

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
    run(f"{ctx.obj.project.Settings.FAImodPath}/point2model  -color 128,0,128 {output_path / 'combined_ampa_poses.txt'} {output_path / 'combined_ampa_poses.mod'}", shell=True)

    # Combine all three models
    tomogram_path= ctx.obj.project_directory.parent / (ctx.obj.relion_tomograms[tomogram.FATomogramName].rlnTomoTiltSeriesName[:-4] + tomogram.FAPL1TomoSuffixForFindbeads)
    run(f"{ctx.obj.project.Settings.FAImodPath}/imodjoin -i {tomogram_path} {output_path / 'combined_aunp.mod'} {output_path / 'colored_contour.mod'} {output_path / 'combined_ampa_poses.mod'} {output_path / 'combined_all.mod'}", shell=True)
    open_command = f"3dmod {tomogram_path} {output_path / 'combined_all.mod'}"

    # Turn AuNPs into objs file
    run(f"imod2obj {output_path / 'combined_aunp.mod'} {output_path / 'combined_aunp.obj'}", shell=True)
    # Make a copy of contour file, mesh it and turn it into an vrml2 files
    run(f"cp {output_path / 'colored_contour.mod'} {output_path / 'colored_contour_mesh.mod'}", shell=True)
    run(f"imodmesh {output_path / 'colored_contour_mesh.mod'}", shell=True)
    run(f"imod2vrml2 {output_path / 'colored_contour_mesh.mod'} {output_path / 'colored_contour_mesh.wrl'}", shell=True)
    return open_command

def prepare_aunp_and_synaptic_membrane_model(
        ctx: FAContext,
        subregion: Pipeline1ClusterRegion,
        tomogram: Tomogram,
        output_path: Path,
        data: Pipeline1Data,
):
    
    output_path = Path("finding_ampa") / "pipeline_data" / "pipeline1" / "subregions" / f"{tomogram.FATomogramName}_{subregion.pl1CluserId}"
    output_path.mkdir(parents=True, exist_ok=True)
    write_aunp_model_file(ctx=ctx,
                            AuNPModelfile=subregion.pl1AuNPModelfile,
                            threshold=subregion.pl1AuNPThreshold,
                            output_path=output_path / "aunp.txt")
    subregion.pl1AuNPThresholdedModel = (output_path / "aunp.txt")
    write_synaptic_contour_model_file(ctx=ctx,
                                        ContourModelfile=data.pl1TomogramResults[tomogram.FATomogramName].pl1SegmentationContourFile,
                                        object_id=subregion.pl1PresynapticObjectId,
                                        output_path=output_path / "presynaptic.txt")
    subregion.pl1PresynapticModel = (output_path / "presynaptic.txt")
    write_synaptic_contour_model_file(ctx=ctx,
                                        ContourModelfile=data.pl1TomogramResults[tomogram.FATomogramName].pl1SegmentationContourFile,
                                        object_id=subregion.pl1PostsynapticObjectId,
                                        output_path=output_path / "postsynaptic.txt")
    subregion.pl1PostsynapticModel = (output_path / "postsynaptic.txt")

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


    

    