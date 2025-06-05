from pathlib import Path

def find_pair_centers(coordinates, distances=(6,10)):
    from scipy.spatial import cKDTree as KDTree
    from scipy.spatial import distance
    import numpy as np
    tree = KDTree(coordinates)
    
    all_pairs = tree.query_pairs(distances[1])
    all_pairs = [pair for pair in all_pairs if distances[0] < distance.euclidean(coordinates[pair[0]], coordinates[pair[1]]) < distances[1]]
    return np.array([[pair[0],pair[1],*((coordinates[pair[0]]+coordinates[pair[1]])/2)] for pair in all_pairs])

def find_closest_point_on_membrane(centers,postsynaptic_coordinates):
    from scipy.spatial import cKDTree as KDTree
    import numpy as np
    postsynaptic_tree = KDTree(postsynaptic_coordinates)
    return np.array([postsynaptic_coordinates[postsynaptic_tree.query(center)[1]] for center in centers])


def estimate_ampa_poses(
        tomo_name,
        aunp_coordinates,
        postsynaptic_data,
        output_filename,
        inter_aunp_distance=(6,10),
        aunp_membrane_distance=(17,23)
):
    import numpy as np

    output_path = Path("ampa_poses")
    output_path.mkdir(exist_ok=True)
    import mrcfile
  
    from scipy.spatial.transform import Rotation as R
    import starfile
    import pandas as pd
    from imodmodel import ImodModel
    from imodmodel.models import Object, Contour, ContourHeader, ObjectHeader

    pairs = find_pair_centers(aunp_coordinates,distances=inter_aunp_distance)
    print(f"Potential pairs after AuNP distance filtering ({inter_aunp_distance}): {len(pairs)}")
    if len(pairs) == 0:
        return
    center_coordinates = pairs[:,2:]
    aunp1_coordinates = aunp_coordinates[pairs[:,0].astype(int)]
    aunp2_coordinates = aunp_coordinates[pairs[:,1].astype(int)]
    closest_points = find_closest_point_on_membrane(center_coordinates, postsynaptic_data)
    closest_points_aunp1 = find_closest_point_on_membrane(aunp1_coordinates, postsynaptic_data)
    closest_points_aunp2 = find_closest_point_on_membrane(aunp2_coordinates, postsynaptic_data)
    aunp1_selection = np.logical_and(aunp_membrane_distance[0] < np.linalg.norm(closest_points_aunp1 - aunp1_coordinates, axis=1), np.linalg.norm(closest_points_aunp1 - aunp1_coordinates, axis=1) < aunp_membrane_distance[1])
    aunp2_selection = np.logical_and(aunp_membrane_distance[0] < np.linalg.norm(closest_points_aunp2 - aunp2_coordinates, axis=1), np.linalg.norm(closest_points_aunp2 - aunp2_coordinates, axis=1) < aunp_membrane_distance[1])
    selection_mask = np.logical_and(aunp1_selection, aunp2_selection)
    pairs = pairs[selection_mask]
    closest_points = closest_points[selection_mask]
    
    print(f"Final pairs after membrane distance filtering ({aunp_membrane_distance}): {len(pairs)}")
    
    ampa_positions = []
    results_relion = []

    for x, (pair, cp) in enumerate(zip(pairs, closest_points)):
        ampa_positions.append([])
        ampa_positions[-1].append({
            "o": 1,
            "c": x+1,
            "x": aunp_coordinates[int(pair[0])][0],
            "y": aunp_coordinates[int(pair[0])][1],
            "z": aunp_coordinates[int(pair[0])][2]
        })
        ampa_positions[-1].append({
            "o": 1,
            "c": x+1,
            "x": aunp_coordinates[int(pair[1])][0],
            "y": aunp_coordinates[int(pair[1])][1],
            "z": aunp_coordinates[int(pair[1])][2]
        })
        ampa_positions[-1].append({
            "o": 1,
            "c": x+1,
            "x": cp[0],
            "y": cp[1],
            "z": cp[2]
        })

        vector_center = pair[2:] - cp
        vector_aunp1 = aunp_coordinates[int(pair[0])] - cp
        vector_aunp2 = aunp_coordinates[int(pair[1])] - cp
        vector_norm = vector_center / np.linalg.norm(vector_center)

        # 60 is fudge factor to line out TMP with membrane
        center = cp + vector_norm * 6
        # Get thre rotation betwen vector and [0,0,1] as a 3D eulerangle

        rot, _ = R.align_vectors([[-0.5, 0, 1],[0.5,0,1]], [vector_aunp1,vector_aunp2])
        eulers = rot.as_euler("ZYZ", degrees=True)

        # Get the binning factor
        
        results_relion.append({
            "rlnTomoName": tomo_name,
            "rlnCoordinateX": center[0],
            "rlnCoordinateY": center[1],
            "rlnCoordinateZ": center[2],

            "rlnAngleRot": eulers[0],
            "rlnAngleTilt": eulers[1],
            "rlnAnglePsi": eulers[2],

            "faAuNPSeperation" : np.linalg.norm(aunp_coordinates[int(pair[0])] - aunp_coordinates[int(pair[1])]),
            "faAuNPMembraneDistance": np.linalg.norm(vector_center),
            # "rlnImagePixelSize": ctx.obj.relion_tomograms[tomogram.FATomogramName].rlnTomoTiltSeriesPixelSize,
            # "rlnMicrographName": "micrograph.mrc",
            # "rlnOpticsGroup": 1,
            # "faVectorX": vector_norm[0],
            # "faVectorY": vector_norm[1],
            # "faVectorZ": vector_norm[2],
        })
    
    #pd.DataFrame(ampa_positions).to_csv(output_path / "ampa_positions.txt", sep=" ", header=None, index=False, mode="w")
    model = ImodModel(
        objects = [
            Object(
                header = ObjectHeader(
                    contsize = len(ampa_positions),
                ),
                contours = [
                    Contour(
                        header = ContourHeader(
                            psize = len(ampa_pose),
                            flags=16,
                            time=0,
                            surf=0
                        ),
                        points = np.concatenate([np.array([p["x"], p["y"], p["z"]]) for p in ampa_pose])
                    ) 
                    for ampa_pose in ampa_positions
                ]
            )
        ]
    )
    model.to_file(output_path / f"{output_filename}.mod")
    print(f"Saved IMOD model of AMPA poses to {output_path / f'{output_filename}.mod'}")
    starfile.write({
        'particles': pd.DataFrame(results_relion), 
        'optics': pd.DataFrame([
            {'rlnOpticsGroup':1}
            ])
    }, output_path / f"{output_filename}.star")
    print(f"Saved RELION star file of AMPA poses to {output_path / f'{output_filename}.star'}")