import imodmodel
from pathlib import Path
import numpy as np
from scipy.spatial import KDTree, distance
from scipy.spatial.transform import Rotation as R
import starfile
import pandas as pd

data_folder = Path("/run/media/elferich/d39abcf3-672c-47a3-85bb-7798f8d15655/elferich/findingampa_data/tmp/find_ampa_poses/")

files_to_analyze = [
    [data_folder / "aunp1.mod", data_folder / "postsynaptic2.mod"],
    [data_folder / "aunp2.mod", data_folder / "postsynaptic3.mod"],
    [data_folder / "aunp3.mod", data_folder / "postsynaptic1.mod"],
    [data_folder / "aunp4.mod", data_folder / "postsynaptic1.mod"],
]

def find_pair_centers(aunp_data):
    coordinates = aunp_data[["x", "y", "z"]].to_numpy() * 10
    tree = KDTree(coordinates)
    raw_pairs = tree.query_pairs(150)  # Find pairs with distance less than 100

    # Sort pairs by distance
    sorted_pairs = sorted(raw_pairs, key=lambda pair: distance.euclidean(coordinates[pair[0]], coordinates[pair[1]]))

    # Ensure pairs are exclusive
    exclusive_pairs = set()
    used_points = set()
    for pair in sorted_pairs:
        if pair[0] not in used_points and pair[1] not in used_points:
            exclusive_pairs.add(pair)
            used_points.update(pair)

    return np.array([(coordinates[pair[0]]+coordinates[pair[1]])/2 for pair in exclusive_pairs])

def find_closest_point_on_membrane(centers,postsynaptic_data):
    postsynaptic_coordinates = postsynaptic_data[["x", "y", "z"]].to_numpy() * 10 
    postsynaptic_tree = KDTree(postsynaptic_coordinates)
    return np.array([postsynaptic_coordinates[postsynaptic_tree.query(center)[1]] for center in centers])


if __name__ == "__main__":
    # Open a txt file to write results to
    with open(data_folder / "results.txt", "w") as results_file:
        gold_scaled = open(data_folder / "gold_scaled.txt", "w")
        results_relion = []
        #results_file.write("Cluster ID\tDistance to Membrane\n")
        for cluster_id,files in enumerate(files_to_analyze):
            aunp_data = imodmodel.read(files[0])
            for i, a in enumerate(aunp_data[["x", "y", "z"]].to_numpy() * 10):
                gold_scaled.write(f"{cluster_id+1} {i+1} {a[0]} {a[1]} {a[2]}\n")
            postsynaptic_data = imodmodel.read(files[1])
            centers = np.array(find_pair_centers(aunp_data))
            closest_points = find_closest_point_on_membrane(centers, postsynaptic_data)
            for x, (c, cp) in enumerate(zip(centers, closest_points)):
                results_file.write(f"{cluster_id+1} {x+1} {c[0]} {c[1]} {c[2]}\n")
                results_file.write(f"{cluster_id+1} {x+1} {cp[0]} {cp[1]} {cp[2]}\n")
                vector = c - cp

                vector_norm = vector / np.linalg.norm(vector)

                cp = cp + vector_norm * 60
                # Get thre rotation betwen vector and [0,0,1] as a 3D eulerangle

                rot, angle = R.align_vectors([[0, 0, 1]], [vector])
                eulers = rot.as_euler("ZYZ", degrees=True)
                results_relion.append({
                    "rlnCoordinateX": cp[0],
                    "rlnCoordinateY": cp[1],
                    "rlnCoordinateZ": cp[2],
                    "rlnOriginX": 0,
                    "rlnOriginY": 0,
                    "rlnOriginZ": 0,
                    "rlnAngleRot": eulers[0],
                    "rlnAngleTilt": eulers[1],
                    "rlnAnglePsi": eulers[2],
                })
        starfile.write(pd.DataFrame(results_relion), data_folder / "relion_poses.star")
        gold_scaled.close()
        