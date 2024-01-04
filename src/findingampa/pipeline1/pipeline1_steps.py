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

def run_findbeads3d(ctx: FAContext, subregion: Pipeline1ClusterRegion, output_path: Path):
    from subprocess import run

    tomogram = next(tomogram for tomogram in ctx.obj.tomograms if tomogram.FATomogramID == subregion.pl1TomogramId)
    beads_model_file_dir = (output_path / "findbeads")
    beads_model_file_dir.mkdir(parents=True, exist_ok=True)
    beads_model_file = beads_model_file_dir / f"{tomogram.FATomogramName}_{subregion.pl1CluserId}.mod"
    beads_point_file = beads_model_file_dir / f"{tomogram.FATomogramName}_{subregion.pl1CluserId}.txt"
    run(f"{ctx.obj.project.Settings.FAImodPath}/findbeads3d -inp {ctx.obj.project.Settings.FADataDir / tomogram.FABin10Tomogram}  -si 1.5 -a -50,50 -yl -thresh 30 -sto 0.5 -o {beads_model_file} -x {subregion.pl1Xmin},{subregion.pl1Xmax} -z {subregion.pl1Zmin},{subregion.pl1Zmax} -yminmax {subregion.pl1Ymin},{subregion.pl1Ymax}", shell=True)
    run(f"{ctx.obj.project.Settings.FAImodPath}/model2point -ob -v -1 {beads_model_file} {beads_point_file}", shell=True)
    subregion.pl1AuNPModelfile = beads_model_file

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

def estimate_ampa_poses(
        ctx: FAContext,
        subregion: Pipeline1ClusterRegion,
):
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
    