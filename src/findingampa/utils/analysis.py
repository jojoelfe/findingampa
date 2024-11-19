import numpy as np

def get_active_zone_points(presynaptic_glb, postsynaptic_glb, distance=28):
    from scipy.spatial import cKDTree
    import pymeshlab

    prems = pymeshlab.MeshSet()
    prems.load_new_mesh(str(presynaptic_glb))
    postms = pymeshlab.MeshSet()
    postms.load_new_mesh(str(postsynaptic_glb))


    print(f"Pre: {prems.mesh_number()} Post: {postms.mesh_number()}")
    prem = prems.current_mesh()
    postm = postms.current_mesh()

    # Magic conversion to nm and correct coordinate system
    pre_points = prem.vertex_matrix().copy()[:,[0,2,1]][::40] * np.array([10,-10,10])
    post_points = postm.vertex_matrix().copy()[:,[0,2,1]][::40] * np.array([10,-10,10])

    pre_normals = prem.vertex_normal_matrix().copy()[:,[0,2,1]][::40] * np.array([10,-10,10])
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


    return(pre_points_in_active_zone, pre_tree, post_poins_in_active_zone, post_tree)

def create_coordinate_system_from_points(presynaptic_points, postsynaptic_points):

    all_poins = np.concatenate([presynaptic_points, postsynaptic_points])

    ap_center = np.mean(all_poins, axis=0)
    u, sigma, v = np.linalg.svd(all_poins-ap_center)
    norm_vector = v[2] / np.linalg.norm(v[2])

    v1 = np.array([0, 1, 0])
    v2 = np.cross(norm_vector, v1)
    v3 = np.cross(norm_vector, v2)
    v1 = v1 / np.linalg.norm(v1) 
    v2 = v2 / np.linalg.norm(v2) 
    v3 = v3 / np.linalg.norm(v3)

    return (np.array([v2, v3, norm_vector]), ap_center)

def active_zonogram(tomogram_path):
    from pathlib import Path
    import mrcfile
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    import torch

    tomo_name = Path(tomogram_path).stem
    tomogram_base = Path(tomogram_path) / "best_alignment"

    with mrcfile.open(list(tomogram_base.glob(f"*_ddw.mrc"))[0]) as mrc:
        tomogram_data_ddw = torch.tensor(mrc.data.copy())

    pre_points, pre_tree, post_points, post_tree = get_active_zone_points(tomogram_base / "aunps" / "presynatpticmembranes.glb", tomogram_base / "aunps" / "postsynapticmembranes.glb")

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
    print(grid.shape)
    if grid.shape[1] > 90000000:
        raise ValueError("Too many points to plot")
    pre_dis, pre_i = pre_tree.query(grid, k=1)
    post_dis, post_i = post_tree.query(grid, k=1)

    grid_selection = np.logical_and(pre_dis < 20, post_dis < 20)

    # Calculate elev, azim, roll to look down cs[2] axis
    elev = np.arcsin(coordinate_system[2][2]) * 180 / np.pi + 90
    azim= np.arctan2(coordinate_system[2][1], coordinate_system[2][0]) * 180 / np.pi
    roll=0

    plt.figure()
    ax = plt.subplot(111, projection='3d')
    # rotate plot
    ax.view_init(elev=elev, azim=azim,roll=roll)
    ax.scatter(post_points[:,0], post_points[:,1], post_points[:,2], c='r',s=0.1,alpha=0.1)
    ax.scatter(pre_points[:,0], pre_points[:,1], pre_points[:,2], c='b', s=0.1,alpha=0.1)

    ax.scatter(ap_center[0], ap_center[1], ap_center[2], c='g', s=10)
    ax.quiver(ap_center[0], ap_center[1], ap_center[2], coordinate_system[2][0]*50, coordinate_system[2][1]*50, coordinate_system[2][2]*50, color='g')
    ax.quiver(ap_center[0], ap_center[1], ap_center[2], coordinate_system[0][0]*50, coordinate_system[0][1]*50, coordinate_system[0][2]*50, color='b')
    ax.quiver(ap_center[0], ap_center[1], ap_center[2], coordinate_system[1][0]*50, coordinate_system[1][1]*50, coordinate_system[1][2]*50, color='r')
    ax.scatter(grid.reshape(-1,3)[::1000,0], grid.reshape(-1,3)[::1000,1], grid.reshape(-1,3)[::1000,2], c=grid_selection[0,::1000], s=0.1)
    plt.show()


    grid_sc = grid / (torch.tensor(tomogram_data_ddw.shape).flip(dims=(0,)).double() / 2.0 ) - 1.0


    grid_sc = grid_sc.reshape(1,2*w+1,2*h+1,2*d+1,3).float()
    tomogram_data_ddw_sc = tomogram_data_ddw.reshape(1,1, *tomogram_data_ddw.shape)
    res_ddw = torch.nn.functional.grid_sample(tomogram_data_ddw_sc, grid_sc,align_corners=False)
    
    res_ddw = res_ddw [0,0]
    rshape = res_ddw.shape
    res_ddw = res_ddw.flatten() * grid_selection
    res_ddw = res_ddw.reshape(rshape)
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 2, width_ratios=[res_ddw.shape[1], res_ddw.shape[2]], height_ratios=[res_ddw.shape[0], res_ddw.shape[2]])

    axxy = plt.subplot(gs[0, 0])
    axxz = plt.subplot(gs[0, 1], sharey=axxy)
    axyz = plt.subplot(gs[1, 0], sharex=axxy)

    axxy.imshow(torch.min(res_ddw,axis=2).values,cmap='gray',interpolation='mitchell',vmax=-0.,vmin=-1.5)
    axxz.imshow(torch.min(res_ddw,axis=1).values,cmap='gray',interpolation='mitchell',vmax=-0.,vmin=-1.5)
    axyz.imshow(torch.min(res_ddw,axis=0).values.T,cmap='gray',interpolation='mitchell',vmax=-0.,vmin=-1.5)
    # Hide axes 
    axxy.axis('off')
    axxz.axis('off')
    axyz.axis('off')
    plt.tight_layout()
    plt.show()
