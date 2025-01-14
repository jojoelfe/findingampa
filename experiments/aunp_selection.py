import streamlit as st

import starfile
from findingampa.utils.analysis import render_active_zonograms, calculate_probit, estimate_anisotropic_gaussian_for_probit, calculate_1q_p_value, render_active_zonograms, select_aunps
import pymeshlab
from pathlib import Path
from scipy.spatial import cKDTree
from matplotlib import pyplot as plt
import torch
import mrcfile
import numpy as np


tom = st.text_input("Enter tomogram", "20240111_WaffleHipp_227")

aunp_data = starfile.read(f"/goliath/processing/Gouaux/CJS/BestTomo/ProcessingCJS/tomograms/15F1_tomograms/TOP_TOMOS/{tom}/best_alignment/aunps/aunp_tm_BP.star")

amp_rank, amp_pro = calculate_probit(aunp_data["faTomo_gaussianFitAmplitude"].to_numpy()* -1)
pea_rank, pea_pro = calculate_probit(aunp_data["faTM_PeakScore"].to_numpy())

Dinv_, Uinv__ = estimate_anisotropic_gaussian_for_probit(amp_pro, pea_pro, None, False, False)
pval,neg_log_p1q_equ_pro_ = calculate_1q_p_value(amp_pro, pea_pro, Dinv_, Uinv__)

sel_aunps = aunp_data[aunp_data["faTomo_gaussianFitAmplitude"] < st.slider("Amplitude", -20.0, 0.0, -6.0)]
sel_aunps = sel_aunps[sel_aunps["faTomo_gaussianFitSigmaX"] < st.slider("SigmaX", 0.0, 5.0, 1.0)]
sel_aunps = sel_aunps[sel_aunps["faTM_PeakScore"] > st.slider("Score", 0.3, 1.0, 0.35)]

with mrcfile.open(f"/goliath/processing/Gouaux/CJS/BestTomo/ProcessingCJS/tomograms/15F1_tomograms/TOP_TOMOS/{tom}/best_alignment/active_zonograms/active_zonogram_0.mrc", permissive=True) as mrc:
    res_ddw = torch.tensor(mrc.data.copy())

az_data = np.load(f"/goliath/processing/Gouaux/CJS/BestTomo/ProcessingCJS/tomograms/15F1_tomograms/TOP_TOMOS/{tom}/best_alignment/active_zonograms/active_zonogram_0.npy" ,allow_pickle=True)

fig = render_active_zonograms((None,None,res_ddw))
selected_aunp_pos = sel_aunps[["faTM_PeakX", "faTM_PeakY", "faTM_PeakZ"]].values + sel_aunps[["faTomo_gaussianFitX", "faTomo_gaussianFitY", "faTomo_gaussianFitZ"]].values

selected_aunp_pos_transformed = (selected_aunp_pos - az_data.tolist()["center"]) @ az_data.tolist()["cs"].T
selected_aunp_pos_transformed += np.floor(np.array(res_ddw.shape)[[2,1,0]]/2)
selected_aunp_pos_transformed = selected_aunp_pos_transformed[np.all(selected_aunp_pos_transformed > 0, axis=1)]
selected_aunp_pos_transformed = selected_aunp_pos_transformed[np.all(selected_aunp_pos_transformed < np.array(res_ddw.shape)[[2,1,0]], axis=1)]
(axxy, axxz, axyz) = fig.get_axes()
axxy.scatter(selected_aunp_pos_transformed[:,0], selected_aunp_pos_transformed[:,1], s=1,c='r')
axxz.scatter(selected_aunp_pos_transformed[:,2], selected_aunp_pos_transformed[:,1], s=1,c='r')
axyz.scatter(selected_aunp_pos_transformed[:,0], selected_aunp_pos_transformed[:,2], s=1,c='r')

st.pyplot(fig)