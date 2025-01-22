import numpy as np

def get_active_zone_points(presynaptic_glb, postsynaptic_glb, distance=28, active_zone=None):
    from scipy.spatial import cKDTree
    import pymeshlab

    prems = pymeshlab.MeshSet()
    prems.load_new_mesh(str(presynaptic_glb))
    postms = pymeshlab.MeshSet()
    postms.load_new_mesh(str(postsynaptic_glb))

    num_potential_active_zones  = prems.mesh_number() * postms.mesh_number()
    print(f"Pre: {prems.mesh_number()} Post: {postms.mesh_number()} Potential active zones: {num_potential_active_zones}")
    # Find all permutations of pre and post meshes
    potential_active_zones = []
    for i in range(prems.mesh_number()):
        for j in range(postms.mesh_number()):
            potential_active_zones.append((i,j))
    print(potential_active_zones)
    found_active_zones = []
    for paz in potential_active_zones:
        
        prems.set_current_mesh(paz[0])
        prem = prems.current_mesh()
        pre_points = prem.vertex_matrix().copy()[:,[0,2,1]][::40] * np.array([10,-10,10])
        pre_normals = prem.vertex_normal_matrix().copy()[:,[0,2,1]][::40] * np.array([10,-10,10])
       
        postms.set_current_mesh(paz[1])
        postm = postms.current_mesh()
        post_points = postm.vertex_matrix().copy()[:,[0,2,1]][::40] * np.array([10,-10,10])
        post_normals = postm.vertex_normal_matrix().copy()[:,[0,2,1]][::40] * np.array([10,-10,10])

        pre_tree = cKDTree(pre_points)
        post_tree = cKDTree(post_points)
        active_zone = pre_tree.query_ball_tree(post_tree, 28)
        pre_points_in_active_zone = []
        post_poins_in_active_zone = []
        for i,a in enumerate(active_zone):
                if len(a) > 0:
                    for psp in a:
                        diff = pre_points[i] - post_points[psp]
                        if np.dot(pre_normals[i], diff) < 0 and np.dot(post_normals[psp], diff) > 0:
                            pre_points_in_active_zone.append(i)
                            post_poins_in_active_zone.append(psp)
                            
                    #pre_points_in_active_zone.append(pre_points[i])
            #np.int(np.concatenate(active_zone))
        post_poins_in_active_zone = post_points[np.unique(np.array(post_poins_in_active_zone).astype(int))]
        pre_points_in_active_zone = pre_points[np.unique(np.array(pre_points_in_active_zone).astype(int))]
        if len(pre_points_in_active_zone) > 0 and len(post_poins_in_active_zone) > 0:
            found_active_zones.append((pre_points_in_active_zone, post_poins_in_active_zone, paz))

    print(f"Return gaz")
    return(found_active_zones)

def create_coordinate_system_from_points(presynaptic_points, postsynaptic_points):
    from scipy.spatial import cKDTree
    all_poins = np.concatenate([presynaptic_points, postsynaptic_points])

    ap_center = np.mean(all_poins, axis=0)
    #u, sigma, v = np.linalg.svd((all_poins-ap_center).T)
    #norm_vector = u[:,2] / np.linalg.norm(u[:,2])
    #norm_vector = np.mean(postsynaptic_points, axis=0) - np.mean(presynaptic_points, axis=0)
    #norm_vector = norm_vector / np.linalg.norm(norm_vector)

    # Get 100 random points in postsynapse
    post_points_sel = postsynaptic_points[np.random.choice(postsynaptic_points.shape[0], 100, replace=False)]
    # Get closest points in presynapse
    pre_dis, pre_i = cKDTree(presynaptic_points).query(post_points_sel)
    pre_points_el = presynaptic_points[pre_i]
    norm_vector = np.mean(post_points_sel - pre_points_el, axis=0)
    norm_vector = norm_vector / np.linalg.norm(norm_vector)
   

    z = np.array([0, 0, 1])
    xp = np.cross(norm_vector, z)
    yp = np.cross(norm_vector, xp)
    xp = xp / np.linalg.norm(xp) 
    yp = yp / np.linalg.norm(yp) 

    return (np.array([xp, yp, norm_vector]), ap_center)

def generate_active_zonograms(tomogram_name, active_zone = None):
    from pathlib import Path
    import mrcfile
    from scipy.spatial import cKDTree
    import torch


    with mrcfile.open(f"{tomogram_name}_full_rec_BP_3DCTF_BIN4_ddw.mrc") as mrc:
        tomogram_data_ddw = torch.tensor(mrc.data.copy())
    active_zones = get_active_zone_points( Path("aunps") / "presynatpticmembranes.glb", Path("aunps") / "postsynapticmembranes.glb",active_zone=active_zone)
    return_data = []
    for i, az in enumerate(active_zones):
        pre_points, post_points, paz = az

        try:
            coordinate_system, ap_center = create_coordinate_system_from_points(pre_points, post_points)
            print(f"Created coordinate system for active zone {i}")
        except Exception as e:
            print(f"Failed to create coordinate system for active zone {i}: {e}")
            continue
        coordinate_system = np.array(coordinate_system)

        all_poins = np.concatenate([pre_points, post_points])
        all_poins_transformed = (torch.tensor(all_poins) - torch.tensor(ap_center)) @ torch.tensor(coordinate_system.T)
        w, h, d = (torch.max(-torch.min(all_poins_transformed, dim=0).values, torch.max(all_poins_transformed, dim=0).values).floor() + 1).int()
        w += 30
        w *= 2
        h += 30
        h *= 2
        d += 10
        d *= 2
        x = torch.arange(0, w-0.99, 1, dtype=torch.float64) - w / 2
        y = torch.arange(0, h-0.99, 1, dtype=torch.float64) - h / 2
        z = torch.arange(0, d-0.99, 1, dtype=torch.float64) - d / 2 
        mg = torch.cartesian_prod(x, y, z)
        grid = mg @ torch.tensor(coordinate_system) + torch.tensor(ap_center)
        if grid.shape[1] > 90000000:
            raise ValueError("Too many points to plot")
        print("Querying Trees")
        pre_dis, pre_i = cKDTree(pre_points).query(grid.cpu().detach(), k=1)
        post_dis, post_i = cKDTree(post_points).query(grid.cpu().detach(), k=1)

        grid_selection = np.logical_and(pre_dis < 40, post_dis < 40)

        grid_sc = grid / (torch.tensor(tomogram_data_ddw.shape).flip(dims=(0,)).double() / 2.0 ) - 1.0


        grid_sc = grid_sc.reshape(1,w,h,d,3).float()
        tomogram_data_ddw_sc = tomogram_data_ddw.reshape(1,1, *tomogram_data_ddw.shape)
        print("Starting grid sample")
        res_ddw = torch.nn.functional.grid_sample(tomogram_data_ddw_sc, grid_sc,align_corners=True)
        
        res_ddw = res_ddw [0,0]
        rshape = res_ddw.shape
        res_ddw = res_ddw.flatten() * grid_selection
        res_ddw = res_ddw.reshape(rshape)
        res_ddw = res_ddw.permute(*torch.arange(res_ddw.ndim - 1, -1, -1))
        return_data.append((coordinate_system, ap_center, res_ddw, paz))
    print("Done gaz")
    return return_data

def read_manual_active_zones(tomogram_name):
    import json
    from pathlib import Path
    import torch
    import mrcfile
    try:
        data = json.loads(Path("fa_manual_override.json").read_text())
    except FileNotFoundError:
        return []
    
    with mrcfile.open(f"{tomogram_name}_full_rec_BP_3DCTF_BIN4_ddw.mrc") as mrc:
        tomogram_data_ddw = torch.tensor(mrc.data.copy())
    
    return_data = []
    for maz in data["manual_active_zones"]:
        center = np.array(maz["center"])
        size = np.array(maz["size"])
        rotation = np.array(maz["rotation"])

        x = torch.arange(0, size[0]-0.99, 1, dtype=torch.float64) - size[0] / 2
        y = torch.arange(0, size[1]-0.99, 1, dtype=torch.float64) - size[1] / 2
        z = torch.arange(0, size[2]-0.99, 1, dtype=torch.float64) - size[2] / 2 
        mg = torch.cartesian_prod(x, y, z)
        grid = mg @ torch.tensor(rotation) + torch.tensor(center)
        if grid.shape[1] > 90000000:
            raise ValueError("Too many points to plot")
       

        grid_sc = grid / (torch.tensor(tomogram_data_ddw.shape).flip(dims=(0,)).double() / 2.0 ) - 1.0


        grid_sc = grid_sc.reshape(1,int(size[0]),int(size[1]),int(size[2]),3).float()
        tomogram_data_ddw_sc = tomogram_data_ddw.reshape(1,1, *tomogram_data_ddw.shape)
        print("Starting grid sample")
        res_ddw = torch.nn.functional.grid_sample(tomogram_data_ddw_sc, grid_sc,align_corners=True)
        
        res_ddw = res_ddw [0,0]
       
        res_ddw = res_ddw.permute(*torch.arange(res_ddw.ndim - 1, -1, -1))
        return_data.append((rotation, center, res_ddw, ()))
    return return_data

def render_active_zonograms(active_zone_data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    import torch

    res_ddw = active_zone_data[2]
    width = (res_ddw.shape[2] + res_ddw.shape[0]) / 50
    height =(res_ddw.shape[1] + res_ddw.shape[0]) / 50
    fig = plt.figure(figsize=(width,height))
    gs = gridspec.GridSpec(2, 2, width_ratios=[res_ddw.shape[2], res_ddw.shape[0]], height_ratios=[res_ddw.shape[1], res_ddw.shape[0]])

    axxy = plt.subplot(gs[0, 0])
    axyz = plt.subplot(gs[0, 1], sharey=axxy)
    axxz = plt.subplot(gs[1, 0], sharex=axxy)
    
    axxy.imshow(torch.min(res_ddw,axis=0).values,cmap='gray',interpolation='mitchell',vmax=-0.,vmin=-20*res_ddw.std(), origin='lower')
    # Draw arrows for coordinate system in xy plane
    axxy.quiver(0,0, 0, 60, color='g', angles='xy', scale_units='xy',units="xy",width=1, scale=1, clip_on=False)
    axxy.quiver(0,0, 60, 0, color='r', angles='xy', scale_units='xy',units="xy",width=1, scale=1, clip_on=False)
    axyz.imshow(torch.min(res_ddw,axis=2).values.T,cmap='gray',interpolation='mitchell',vmax=-0.,vmin=-20*res_ddw.std(), origin='lower')
    axyz.quiver(0,0, 0, 60, color='g', angles='xy', scale_units='xy',units="xy",width=1, scale=1, clip_on=False)
    axyz.quiver(0,0, 60, 0, color='b', angles='xy', scale_units='xy',units="xy",width=1, scale=1, clip_on=False)
    axxz.imshow(torch.min(res_ddw,axis=1).values,cmap='gray',interpolation='mitchell',vmax=-0.,vmin=-20*res_ddw.std(), origin='lower')
    axxz.quiver(0,0, 0, 60, color='b', angles='xy', scale_units='xy',units="xy",width=1, scale=1, clip_on=False)
    axxz.quiver(0,0, 60, 0, color='r', angles='xy', scale_units='xy',units="xy",width=1, scale=1, clip_on=False)
    # Hide axes 
    axxy.axis('off')
    axxz.axis('off')
    axyz.axis('off')
    plt.tight_layout()
    return fig
        

def render_active_zone_extend_via_blender(blenderfile, center, size, rotation_matrix, output_filename):
    import bpy
    import numpy as np
    from pathlib import Path
    import mathutils
    import os
    from findingampa.utils.assets import FA_ASSETS

    bpy.ops.wm.open_mainfile(filepath=str(blenderfile))

    bm = mathutils.Matrix(rotation_matrix.T)

    bpy.ops.mesh.primitive_cube_add(size=1, location=center / 10.0, scale=size / 10.0, rotation=bm.to_euler())

    cube = bpy.context.active_object
    # Apply location, rotation and scale
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bevel_modifier = cube.modifiers.new(name="Wireframe", type='WIREFRAME')
    bevel_modifier.thickness = 0.5



    # Append from blend file
    bpy.ops.wm.append("EXEC_DEFAULT",directory=os.path.join(str(FA_ASSETS / "blender" / "coordinatesystem.blend"),"Object"),filename="CoordinateSystem",use_recursive=True)

    coordinate_system = bpy.data.objects["CoordinateSystem"]
    coordinate_system.location =  (-size / 20.0)
    
    # Set coordinate_system as the active object
    bpy.context.view_layer.objects.active = coordinate_system
    # Apply modifier
    bpy.ops.object.modifier_apply(modifier="GeometryNodes")
    bpy.ops.object.transform_apply(location=True)
    coordinate_system.location = center / 10.0
    coordinate_system.rotation_euler = bm.to_euler()
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.samples = 50
    bpy.ops.render.render()

    bpy.data.images['Render Result'].save_render(filepath=str(output_filename))


    



def find_surfaces(coors):
    # Use RANSAC to fit a plane to the data
    from sklearn.linear_model import RANSACRegressor
    from sklearn.linear_model import LinearRegression
    import numpy as np

    ransac = RANSACRegressor(LinearRegression(), residual_threshold=1, max_trials=1000)
    ransac.fit(coors[:,1:3], coors[:,0])

    print(f"PLane 1 {ransac.estimator_.coef_, ransac.estimator_.intercept_}")

    # Get all points within 10 of plane
    near_mask = np.abs(coors[:,0] - ransac.predict(coors[:,1:3])) < 5
    center = np.array([375,500,500]) // 2
    away_mask = np.sign(coors[:,0] - ransac.predict(coors[:,1:3])) != np.sign(center[0]- ransac.predict(np.expand_dims(center[1:3],0)))
    sel_mask = np.logical_or(near_mask,away_mask)

    new_coors = coors[~sel_mask]
    ransac2 = RANSACRegressor(LinearRegression(), residual_threshold=1, max_trials=1000)
    ransac2.fit(new_coors[:,1:3], new_coors[:,0])

    print(f"Plane 2:{ransac2.estimator_.coef_, ransac2.estimator_.intercept_}")

    # Get all points within 5 of plane
    near_mask = np.abs(coors[:,0] - ransac2.predict(coors[:,1:3])) < 5
    center = np.array([375,500,500]) // 2
    away_mask = np.sign(coors[:,0] - ransac2.predict(coors[:,1:3])) != np.sign(center[0]- ransac2.predict(np.expand_dims(center[1:3],0)))
    sel_mask2 = np.logical_or(near_mask,away_mask)

    return (sel_mask, sel_mask2)


from scipy.special import erfinv, erf
import matplotlib.colors as mcolors
import math
def rho_g2(x,y,C):
    return 1/(2*np.pi) * np.sqrt(np.linalg.det(C)) * np.exp(-0.5 * (x**2*C[0,0] + x*y*(C[0,1] + C[1,0]) + y**2*C[1,1]))
    # when x is 129*128 and y is 129*128, can also be calculated as 1) a = np.array([x.flatten(),y.flatten()]) 2) np.diag(np.matmul(np.matmul(a.T, C),a)) 3) reshape

def calculate_probit(x_):
    # calculate rank and remap to quantile of standard gaussian
    rank_x_ = np.argsort(np.argsort(x_))+1 # use argsort twice to ensure the ranking is correct
    rank_x_ = (rank_x_ - 0.5) / max(1, len(x_))
    pro_x_ = np.sqrt(2) * erfinv(2 * rank_x_ - 1.0)
    return rank_x_, pro_x_

def estimate_anisotropic_gaussian_for_probit(pro_x1_, pro_x2_, output_name=None, save_plot=False, show_plot=True):
    pro_lim_ = [-4.5, 4.5]
    # define anisotropic-gaussian
    n_pro_x = 128
    n_pro_y = n_pro_x+1
    pro_x_ = np.linspace(min(pro_lim_), max(pro_lim_), n_pro_x)
    pro_dx = np.mean(np.diff(pro_x_))

    pro_y_ = np.linspace(min(pro_lim_), max(pro_lim_), n_pro_y)
    pro_dy = np.mean(np.diff(pro_y_))

    pro_x__, pro_y__ = np.meshgrid(pro_x_,pro_y_)

    n_r = len(pro_x1_)
    tmp_ = np.array([pro_x1_, pro_x2_])
    Cinv__ = np.matmul(tmp_, tmp_.T)/n_r
    Dinv_,Uinv__ = np.linalg.eig(Cinv__) # eigenvalues not necessarily ordered
    

    return Dinv_, Uinv__ #C__, g__, pro_x__,pro_y__, pro_dx, pro_dy

def calculate_1q_p_value(ag_x1_, ag_x2_, Dinv_, Uinv__):
    # anisotropic gaussian x1 and x2
    # calculate p-value with a 1-quadrant constraint
    n_r = len(ag_x1_)
    p1q_equ_r_ = np.zeros(n_r)
    tmp_a = max(np.sqrt(Dinv_))
    tmp_b = min(np.sqrt(Dinv_))
    #print("Dinv = [", Dinv_[0], ", ", Dinv_[1], "]")
    #print("first eig-vec = ", Uinv__[:,0])
    #print("second eig-vec = ", Uinv__[:,1])
    if Dinv_[0]<=Dinv_[1]:
        Uinv__ = np.array([Uinv__[:,1], Uinv__[:,0]]).T
    if (Uinv__[1,0] < 0) and (Uinv__[0,0]<0):
        Uinv__[1,0]*=-1
        Uinv__[0,0]*=-1
    tmp_w = math.atan2(Uinv__[1,0], Uinv__[0,0])
    #print("Major axis = "+str(tmp_w*180/np.pi)+" deg \n\n")
    tmp_gamma = math.atan2(1.0, 0.5*np.sin(2*tmp_w)*(tmp_b/tmp_a-tmp_a/tmp_b)) # angular formula only valid in 2-dimensions
    for i in range(0, n_r):
        ag_x1 = ag_x1_[i]
        ag_x2 = ag_x2_[i]
        p1q_equ = 1.0
        if (ag_x1>0) and (ag_x2>0):
            tmp_x_0 = +np.cos(tmp_w)*ag_x1 + np.sin(tmp_w)*ag_x2 
            tmp_x_1 = -np.sin(tmp_w)*ag_x1 + np.cos(tmp_w)*ag_x2 # R*S: [cos(w) -sin(w), sin(w) cos(w)]*[sx 0, 0 sy] rotation matrix * scaling matrix
            tmp_y_0 = tmp_x_0/max(1e-12, tmp_a)
            tmp_y_1 = tmp_x_1/max(1e-12, tmp_b)
            tmp_y_r = np.sqrt(tmp_y_0**2+tmp_y_1**2)
            p1q_equ = np.exp(-tmp_y_r**2/2)*tmp_gamma/(2*np.pi)
        p1q_equ_r_[i] = p1q_equ

    return p1q_equ_r_,-np.log(p1q_equ_r_)

def select_aunps(aunp_data, az_data, res_ddw, threshold = 3.8):
    from findingampa.utils.analysis import render_active_zonograms
    import pymeshlab
    from pathlib import Path
    from scipy.spatial import cKDTree
    import json


    try:
        data = json.loads(Path("fa_manual_override.json").read_text())
    except FileNotFoundError:
        data = {}
    if "aunp_thresholds" in data:
        print("Using manual thresholds")
        significant_picks_mask = np.logical_and(
            aunp_data["faTomo_gaussianFitAmplitude"] < data["aunp_thresholds"]["amplitude"],
            aunp_data["faTomo_gaussianFitSigmaX"] < data["aunp_thresholds"]["sigmax"],
            aunp_data["faTM_PeakScore"] > data["aunp_thresholds"]["score"]
        )
    else:
        amp_rank, amp_pro = calculate_probit(aunp_data["faTomo_gaussianFitAmplitude"].to_numpy()* -1)
        pea_rank, pea_pro = calculate_probit(aunp_data["faTM_PeakScore"].to_numpy())

        Dinv_, Uinv__ = estimate_anisotropic_gaussian_for_probit(amp_pro, pea_pro, None, False, False)
        pval,neg_log_p1q_equ_pro_ = calculate_1q_p_value(amp_pro, pea_pro, Dinv_, Uinv__)
        significant_picks_mask = neg_log_p1q_equ_pro_ > threshold
        


    fig = render_active_zonograms((None,None,res_ddw))


    
    sel_aunps = aunp_data[significant_picks_mask]

    selected_aunp_pos = sel_aunps[["faTM_PeakX", "faTM_PeakY", "faTM_PeakZ"]].values + sel_aunps[["faTomo_gaussianFitX", "faTomo_gaussianFitY", "faTomo_gaussianFitZ"]].values

    if len(az_data.tolist()['objects']) < 0:
        presynaptic_glb =  Path("aunps") / "presynatpticmembranes.glb"
        postsynaptic_glb =  Path("aunps") / "postsynapticmembranes.glb"
        prems = pymeshlab.MeshSet()
        prems.load_new_mesh(str(presynaptic_glb))
        postms = pymeshlab.MeshSet()
        postms.load_new_mesh(str(postsynaptic_glb))

        prems.set_current_mesh(az_data.tolist()['objects'][0])
        prem = prems.current_mesh()
        pre_points = prem.vertex_matrix().copy()[:,[0,2,1]][::40] * np.array([10,-10,10])
        pre_normals = prem.vertex_normal_matrix().copy()[:,[0,2,1]][::40] * np.array([10,-10,10])
        postms.set_current_mesh(az_data.tolist()['objects'][1])
        postm = postms.current_mesh()
        post_points = postm.vertex_matrix().copy()[:,[0,2,1]][::40] * np.array([10,-10,10])
        post_normals = postm.vertex_normal_matrix().copy()[:,[0,2,1]][::40] * np.array([10,-10,10])
        pre_tree = cKDTree(pre_points)
        post_tree = cKDTree(post_points)



        pre_dis, pre_idx = pre_tree.query(selected_aunp_pos)
        post_dis, post_idx = post_tree.query(selected_aunp_pos)

        selected_aunp_pos_mask = np.logical_and(pre_dis < 30, post_dis < 30)
        selected_aunp_pos_postsyn_mask = np.logical_and(post_dis < 30, np.logical_not(selected_aunp_pos_mask))
    else:
        selected_aunp_pos_mask = np.ones(selected_aunp_pos.shape[0], dtype=bool)
        selected_aunp_pos_postsyn_mask = np.zeros(selected_aunp_pos.shape[0], dtype=bool)
    #selected_aunp_pos_postsyn_mask = np.logical_and(selected_aunp_pos_postsyn_mask,
    # For this to work whe have to select the outer leaflet of the postsynaptic membrane
    #dot_mask = np.einsum('ij,ij->i', post_normals[post_idx], selected_aunp_pos-post_points[post_idx]) > 0
    #print(np.sum(dot_mask),len(dot_mask))
    selected_aunp_pos_masked = selected_aunp_pos[selected_aunp_pos_mask]
    selected_aunp_pos_post = selected_aunp_pos[selected_aunp_pos_postsyn_mask]
    selected_aunp_pos_transformed = (selected_aunp_pos_masked - az_data.tolist()["center"]) @ az_data.tolist()["cs"].T
    selected_aunp_pos_post_transformed = (selected_aunp_pos_post - az_data.tolist()["center"]) @ az_data.tolist()["cs"].T  
    selected_aunp_pos_transformed += np.floor(np.array(res_ddw.shape)[[2,1,0]]/2)
    selected_aunp_pos_post_transformed += np.floor(np.array(res_ddw.shape)[[2,1,0]]/2)
    # Make sure the points are within the volume (remove them if not)
    selected_aunp_pos_transformed = selected_aunp_pos_transformed[np.all(selected_aunp_pos_transformed > 0, axis=1)]
    selected_aunp_pos_transformed = selected_aunp_pos_transformed[np.all(selected_aunp_pos_transformed < np.array(res_ddw.shape)[[2,1,0]], axis=1)]
    selected_aunp_pos_post_transformed = selected_aunp_pos_post_transformed[np.all(selected_aunp_pos_post_transformed > 0, axis=1)]
    selected_aunp_pos_post_transformed = selected_aunp_pos_post_transformed[np.all(selected_aunp_pos_post_transformed < np.array(res_ddw.shape)[[2,1,0]], axis=1)]

    # Get subplot from fig
    (axxy, axxz, axyz) = fig.get_axes()
    axxy.scatter(selected_aunp_pos_transformed[:,0], selected_aunp_pos_transformed[:,1], s=1,c='r')
    axxy.scatter(selected_aunp_pos_post_transformed[:,0], selected_aunp_pos_post_transformed[:,1], s=1,c='g')
    axxz.scatter(selected_aunp_pos_transformed[:,2], selected_aunp_pos_transformed[:,1], s=1,c='r')
    axxz.scatter(selected_aunp_pos_post_transformed[:,2], selected_aunp_pos_post_transformed[:,1], s=1,c='g')
    axyz.scatter(selected_aunp_pos_transformed[:,0], selected_aunp_pos_transformed[:,2], s=1,c='r')
    axyz.scatter(selected_aunp_pos_post_transformed[:,0], selected_aunp_pos_post_transformed[:,2], s=1,c='g')
    return fig, significant_picks_mask, selected_aunp_pos, selected_aunp_pos_mask, selected_aunp_pos_postsyn_mask
    #plt.show()




