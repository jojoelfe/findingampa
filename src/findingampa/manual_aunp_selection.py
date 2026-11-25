import streamlit as st

import starfile
from findingampa.utils.analysis import render_active_zonograms, calculate_probit, estimate_anisotropic_gaussian_for_probit, calculate_1q_p_value, render_active_zonograms, select_aunps
from findingampa.utils.tm3d import refine_aunp_peaks
import pymeshlab
from pathlib import Path
from scipy.spatial import cKDTree
from matplotlib import pyplot as plt
import torch
import mrcfile
import numpy as np
import json

try:
    tomo_name = list(Path.cwd().glob("*.rawtlt"))[0].stem
except IndexError:
    typer.echo("No rawtlt file found")
    raise typer.Exit()

aunp_data = starfile.read(f"aunps/aunp_tm_BP.star")
#aunp_data = refine_aunp_peaks(aunp_data)

try:
    data = json.loads(Path("fa_manual_override.json").read_text())
except FileNotFoundError:
    data = {}

if "aunp_thresholds" in data:
    amplitude_thresh = data["aunp_thresholds"]["amplitude"]
    sigmax_thresh = data["aunp_thresholds"]["sigmax"]
    score_thresh = data["aunp_thresholds"]["score"]
else:
    amplitude_thresh = -1.0
    sigmax_thresh = 1.0
    score_thresh = 0.35
    data["aunp_thresholds"] = {}
if "plane1_elimination" in data["aunp_thresholds"]:
    plane1_elimination = data["aunp_thresholds"]["plane1_elimination"]
    plane1_distance_cutoff = data["aunp_thresholds"]["plane1_distance_cutoff"]
else:  
    plane1_elimination = False
    plane1_distance_cutoff = 5.0
if "plane2_elimination" in data["aunp_thresholds"]:
    plane2_elimination = data["aunp_thresholds"]["plane2_elimination"]
    plane2_distance_cutoff = data["aunp_thresholds"]["plane2_distance_cutoff"]
else:
    plane2_elimination = False
    plane2_distance_cutoff = 5.0

@st.cache_resource
def load_tomogram(az):
    with mrcfile.open(az, permissive=True) as mrc:
        res_ddw = torch.tensor(mrc.data.copy())
    
    return res_ddw, res_ddw.shape

amplitude_thresh = st.slider("Amplitude", -20.0, 0.0, amplitude_thresh)
sigmax_thresh = st.slider("SigmaX", 0.0, 5.0, sigmax_thresh)
score_thresh = st.slider("Score", 0.30, 1.0, score_thresh)
plane1_elimination = st.checkbox("Eliminate plane 1", value=plane1_elimination)
if plane1_elimination:
    plane1_distance_cutoff = st.slider("Plane 1 distance cutoff", 0.0, 100.0, plane1_distance_cutoff)
plane2_elimination = st.checkbox("Eliminate plane 2", value=plane2_elimination)
if plane2_elimination:
    plane2_distance_cutoff = st.slider("Plane 2 distance cutoff", 0.0, 100.0, plane2_distance_cutoff)
sel_aunps = aunp_data[aunp_data["faTomo_gaussianFitAmplitude"] < amplitude_thresh]
sel_aunps = sel_aunps[sel_aunps["faTomo_gaussianFitSigmaX"] < sigmax_thresh]
sel_aunps = sel_aunps[sel_aunps["faTM_PeakScore"] > score_thresh]
if plane1_elimination:
    sel_aunps = sel_aunps[sel_aunps["faTomo_Surface1"] > plane1_distance_cutoff]
if plane2_elimination:
    sel_aunps = sel_aunps[sel_aunps["faTomo_Surface2"] > plane2_distance_cutoff]

for az in Path("active_zonograms").glob("active_zonogram_*.mrc"):
    res_ddw, rshape = load_tomogram(az)
    fig = render_active_zonograms((None,None,res_ddw))
    az_data = np.load(az.with_suffix(".npy") ,allow_pickle=True)

    selected_aunp_pos = sel_aunps[["faTM_PeakX", "faTM_PeakY", "faTM_PeakZ"]].values + sel_aunps[["faTomo_gaussianFitX", "faTomo_gaussianFitY", "faTomo_gaussianFitZ"]].values

    selected_aunp_pos_transformed = (selected_aunp_pos - az_data.tolist()["center"]) @ az_data.tolist()["cs"].T
    selected_aunp_pos_transformed += np.floor(np.array(rshape)[[2,1,0]]/2)
    selected_aunp_pos_transformed = selected_aunp_pos_transformed[np.all(selected_aunp_pos_transformed > 0, axis=1)]
    selected_aunp_pos_transformed = selected_aunp_pos_transformed[np.all(selected_aunp_pos_transformed < np.array(rshape)[[2,1,0]], axis=1)]
    (axxy, axxz, axyz) = fig.get_axes()
    axxy.scatter(selected_aunp_pos_transformed[:,0], selected_aunp_pos_transformed[:,1], s=1,c='r')
    axxz.scatter(selected_aunp_pos_transformed[:,2], selected_aunp_pos_transformed[:,1], s=1,c='r')
    axyz.scatter(selected_aunp_pos_transformed[:,0], selected_aunp_pos_transformed[:,2], s=1,c='r')

    st.pyplot(fig)

save = st.button("Save")
if save:
    import json
    try:
        data = json.loads(Path("fa_manual_override.json").read_text())
    except FileNotFoundError:
        data = {}
    
    data["aunp_thresholds"] = {
        "amplitude": amplitude_thresh,
        "sigmax": sigmax_thresh,
        "score": score_thresh,
        "plane1_elimination": plane1_elimination,
        "plane2_elimination": plane2_elimination,
        "plane1_distance_cutoff": plane1_distance_cutoff if plane1_elimination else 0.0,
        "plane2_distance_cutoff": plane2_distance_cutoff if plane2_elimination else 0.0,
    }
    Path("fa_manual_override.json").write_text(json.dumps(data, indent=4))