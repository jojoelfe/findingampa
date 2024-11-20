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
    
    if active_zone is None:
        prem = prems.current_mesh()
        postm = postms.current_mesh()

        # Magic conversion to nm and correct coordinate system
        pre_points = prem.vertex_matrix().copy()[:,[0,2,1]][::40] * np.array([10,-10,10])
        post_points = postm.vertex_matrix().copy()[:,[0,2,1]][::40] * np.array([10,-10,10])

        pre_normals = prem.vertex_normal_matrix().copy()[:,[0,2,1]][::40] * np.array([10,-10,10])
        post_normals = postm.vertex_normal_matrix().copy()[:,[0,2,1]][::40] * np.array([10,-10,10])
    else:
        prep = []
        pren = []
        for pm in active_zone["pre"]:
            prems.set_current_mesh(pm)
            prem = prems.current_mesh()
            pre_points = prem.vertex_matrix().copy()[:,[0,2,1]][::40] * np.array([10,-10,10])
            prep.append(pre_points)
            pre_normals = prem.vertex_normal_matrix().copy()[:,[0,2,1]][::40] * np.array([10,-10,10])
            pren.append(pre_normals)
        pre_points = np.concatenate(prep)
        pre_normals = np.concatenate(pren)
        postp = []
        postn = []
        for pm in active_zone["post"]:
            postms.set_current_mesh(pm)
            postm = postms.current_mesh()
            post_points = postm.vertex_matrix().copy()[:,[0,2,1]][::40] * np.array([10,-10,10])
            postp.append(post_points)
            post_normals = postm.vertex_normal_matrix().copy()[:,[0,2,1]][::40] * np.array([10,-10,10])
            postn.append(post_normals)

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


    return(pre_points_in_active_zone, post_poins_in_active_zone)

def create_coordinate_system_from_points(presynaptic_points, postsynaptic_points):
    from scipy.spatial import cKDTree
    all_poins = np.concatenate([presynaptic_points, postsynaptic_points])

    ap_center = np.mean(all_poins, axis=0)
    #u, sigma, v = np.linalg.svd((all_poins-ap_center).T)
    #norm_vector = u[:,2] / np.linalg.norm(u[:,2])
    #norm_vector = np.mean(postsynaptic_points, axis=0) - np.mean(presynaptic_points, axis=0)
    #norm_vector = norm_vector / np.linalg.norm(norm_vector)

    # GEt 100 radnom points in postsynapse
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

def active_zonogram(tomogram_path, active_zone = None):
    from pathlib import Path
    import mrcfile
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from scipy.spatial import cKDTree
    import torch

    tomo_name = Path(tomogram_path).stem
    tomogram_base = Path(tomogram_path) / "best_alignment"

    with mrcfile.open(list(tomogram_base.glob(f"*_ddw.mrc"))[0]) as mrc:
        tomogram_data_ddw = torch.tensor(mrc.data.copy())

    pre_points, post_points = get_active_zone_points(tomogram_base / "aunps" / "presynatpticmembranes.glb", tomogram_base / "aunps" / "postsynapticmembranes.glb",active_zone=active_zone)
    return
    plt.figure()
    plt.figsize=(10,10)
    ax = plt.subplot(111, projection='3d')
    # rotate plot
    ax.view_init(elev=90, azim=0,roll=0)
    #ax.set_proj_type('ortho')
    ax.scatter(post_points[:,0], post_points[:,1], post_points[:,2], c='r',s=0.5,alpha=0.1)
    ax.scatter(pre_points[:,0], pre_points[:,1], pre_points[:,2], c='b', s=0.5,alpha=0.1)
    plt.show()
    coordinate_system, ap_center = create_coordinate_system_from_points(pre_points, post_points)
    coordinate_system = np.array(coordinate_system)

    all_poins = np.concatenate([pre_points, post_points])
    all_poins_transformed = (torch.tensor(all_poins) - torch.tensor(ap_center)) @ torch.tensor(coordinate_system.T)
    w, h, d = (torch.max(-torch.min(all_poins_transformed, dim=0).values, torch.max(all_poins_transformed, dim=0).values).floor() + 1).int()
    
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

   
    plt.figure()
    plt.figsize=(10,10)
    ax = plt.subplot(111, projection='3d')
    # rotate plot
    ax.view_init(elev=90, azim=0,roll=0)
    #ax.set_proj_type('ortho')
    ax.scatter(post_points[:,0], post_points[:,1], post_points[:,2], c='r',s=0.5,alpha=0.1)
    ax.scatter(pre_points[:,0], pre_points[:,1], pre_points[:,2], c='b', s=0.5,alpha=0.1)

    ax.scatter(ap_center[0], ap_center[1], ap_center[2], c='g', s=10)
    ax.quiver(ap_center[0], ap_center[1], ap_center[2], coordinate_system[2][0]*50, coordinate_system[2][1]*50, coordinate_system[2][2]*50, color='g')
    ax.quiver(ap_center[0], ap_center[1], ap_center[2], coordinate_system[0][0]*50, coordinate_system[0][1]*50, coordinate_system[0][2]*50, color='b')
    ax.quiver(ap_center[0], ap_center[1], ap_center[2], coordinate_system[1][0]*50, coordinate_system[1][1]*50, coordinate_system[1][2]*50, color='r')
    gr = grid.reshape(2*w+1,2*h+1,2*d+1,3)
    ax.scatter(gr[:,h,:,0], gr[:,h,:,1], gr[:,h,:,2], c=grid_selection.reshape(2*w+1,2*h+1,2*d+1)[:,h,:], s=0.1)
    plt.show()
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
    print(res_ddw.shape)
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 2, width_ratios=[res_ddw.shape[2], res_ddw.shape[0]], height_ratios=[res_ddw.shape[1], res_ddw.shape[0]])

    axxy = plt.subplot(gs[0, 0])
    axxz = plt.subplot(gs[0, 1], sharey=axxy)
    axyz = plt.subplot(gs[1, 0], sharex=axxy)
    print(torch.min(res_ddw,axis=0).values.shape)
    print(res_ddw.mean())
    print(res_ddw.std())
    
    axxy.imshow(torch.min(res_ddw,axis=0).values,cmap='gray',interpolation='mitchell',vmax=-0.,vmin=-20*res_ddw.std())
    axxz.imshow(torch.min(res_ddw,axis=2).values.T,cmap='gray',interpolation='mitchell',vmax=-0.,vmin=-20*res_ddw.std())
    axyz.imshow(torch.min(res_ddw,axis=1).values,cmap='gray',interpolation='mitchell',vmax=-0.,vmin=-20*res_ddw.std())
    # Hide axes 
    axxy.axis('off')
    axxz.axis('off')
    axyz.axis('off')
    plt.tight_layout()
    plt.show()
