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
        #plt.figure()
        #plt.figsize=(10,10)
        #ax = plt.subplot(111, projection='3d')
        # rotate plot
        #ax.view_init(elev=90, azim=0,roll=0)
        #ax.set_proj_type('ortho')
        #ax.scatter(post_points[:,0], post_points[:,1], post_points[:,2], c='g',s=0.5,alpha=0.1)
        #ax.scatter(pre_points[:,0], pre_points[:,1], pre_points[:,2], c='r', s=0.5,alpha=0.1)
        #plt.show()
        try:
            coordinate_system, ap_center = create_coordinate_system_from_points(pre_points, post_points)
        except Exception as e:
            print(f"Failed to create coordinate system for active zone {i}: {e}")
            continue
        coordinate_system = np.array(coordinate_system)

        all_poins = np.concatenate([pre_points, post_points])
        all_poins_transformed = (torch.tensor(all_poins) - torch.tensor(ap_center)) @ torch.tensor(coordinate_system.T)
        w, h, d = (torch.max(-torch.min(all_poins_transformed, dim=0).values, torch.max(all_poins_transformed, dim=0).values).floor() + 1).int()
        w += 30
        h += 30
        d += 10
        x = torch.linspace(-w, w, 2*w+1, dtype=torch.float64)
        y = torch.linspace(-h, h, 2*h+1, dtype=torch.float64)
        z = torch.linspace(-d, d, 2*d+1, dtype=torch.float64)
        mg = torch.cartesian_prod(x, y, z)
        # Construct 100x 100 mesh around pre points center in fractional coordinates using v2 and v3
        grid = mg @ torch.tensor(coordinate_system) + torch.tensor(ap_center)
        if grid.shape[1] > 90000000:
            raise ValueError("Too many points to plot")
        pre_dis, pre_i = cKDTree(pre_points).query(grid, k=1)
        post_dis, post_i = cKDTree(post_points).query(grid, k=1)

        grid_selection = np.logical_and(pre_dis < 40, post_dis < 40)

    
        #plt.figure()
        #plt.figsize=(10,10)
        #ax = plt.subplot(111, projection='3d')
        ## rotate plot
        #ax.view_init(elev=90, azim=0,roll=0)
        #ax.set_proj_type('ortho')
        #ax.scatter(post_points[:,0], post_points[:,1], post_points[:,2], c='g',s=0.5,alpha=0.1)
        #ax.scatter(pre_points[:,0], pre_points[:,1], pre_points[:,2], c='r', s=0.5,alpha=0.1)

        #ax.scatter(ap_center[0], ap_center[1], ap_center[2], c='g', s=10)
        #ax.quiver(ap_center[0], ap_center[1], ap_center[2], coordinate_system[2][0]*50, coordinate_system[2][1]*50, coordinate_system[2][2]*50, color='g')
        #ax.quiver(ap_center[0], ap_center[1], ap_center[2], coordinate_system[0][0]*50, coordinate_system[0][1]*50, coordinate_system[0][2]*50, color='b')
        #ax.quiver(ap_center[0], ap_center[1], ap_center[2], coordinate_system[1][0]*50, coordinate_system[1][1]*50, coordinate_system[1][2]*50, color='r')
        #gr = grid.reshape(2*w+1,2*h+1,2*d+1,3)
        #ax.scatter(gr[:,h,:,0], gr[:,h,:,1], gr[:,h,:,2], c=grid_selection.reshape(2*w+1,2*h+1,2*d+1)[:,h,:], s=0.1)
        #plt.show()
        #return coordinate_system, ap_center

        grid_sc = grid / (torch.tensor(tomogram_data_ddw.shape).flip(dims=(0,)).double() / 2.0 ) - 1.0


        grid_sc = grid_sc.reshape(1,2*w+1,2*h+1,2*d+1,3).float()
        tomogram_data_ddw_sc = tomogram_data_ddw.reshape(1,1, *tomogram_data_ddw.shape)
        res_ddw = torch.nn.functional.grid_sample(tomogram_data_ddw_sc, grid_sc,align_corners=False)
        
        res_ddw = res_ddw [0,0]
        rshape = res_ddw.shape
        res_ddw = res_ddw.flatten() * grid_selection
        res_ddw = res_ddw.reshape(rshape)
        res_ddw = res_ddw.permute(*torch.arange(res_ddw.ndim - 1, -1, -1))
        return_data.append((coordinate_system, ap_center, res_ddw, paz))
    return return_data

def render_active_zonograms(active_zone_data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    import torch

    res_ddw = active_zone_data[2]
    fig = plt.figure()
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


    
