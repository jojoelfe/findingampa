import torch 

def load_relion_matrices(tilt_info, device):
    import numpy as np
    matrices = tilt_info.iloc[:,[0,1,2,3]].to_numpy()
    matrices = [[s.strip('[]').split(',') for s in matrix] for matrix in matrices]
    matrices = np.array([np.vstack([[float(x) for x in row] for row in matrix]) for matrix in matrices])
    matrices = torch.tensor(matrices,device=device)
    return matrices

def relion_matrix_to_xf_and_tilt(matrix: torch.tensor):
    """Convert a relion matrix to a transformation matrix and a tilt angle"""
    from libtilt.transformations import Rx, Rz, T, Ry
    t = T(matrix[:, 0])
    ry = Ry(matrix[:, 1], zyx=False)
    rz = Rz(matrix[:, 2], zyx=False)
    return t @ rz @ ry, matrix[:, 3]

def aunp_coordinates_to_tiltseries(aunp_coordinates, tilt_angles, tilt_axes_angles, translations, tomo_dim, ali_dim, orig_dim, x_offset=0.0, y_offset=0.0, z_offset=0.5):
    from scipy.spatial.transform import Rotation as R
    import numpy as np

    rotations = R.from_euler('Y', tilt_angles, degrees=True)
    print(tomo_dim)
    aunp_coordinates_to_center = aunp_coordinates - [tomo_dim[0]/2-x_offset, tomo_dim[1]/2-y_offset, tomo_dim[2]/2-z_offset]
    aunp_coordinates_rotated = np.array([rotations.apply(a) for a in aunp_coordinates_to_center])
    aunp_coordinates_rotated += [tomo_dim[0]/2-x_offset, tomo_dim[1]/2-y_offset, tomo_dim[2]/2-z_offset]

    rotations = R.from_euler('Z', tilt_axes_angles, degrees=True)
    
    aunp_coordinates_rotated_to_center = aunp_coordinates_rotated - [ali_dim[0]/2, ali_dim[1]/2,0]
    aunp_coordinates_rotated_to_center *= 4
    aunp_coordinates_rotated_to_center[:,:,[0,1]] -= translations
    aunp_coordinates_rotated_rotated = np.array([rotations.inv().apply(a) for a in aunp_coordinates_rotated_to_center])
    aunp_coordinates_rotated_rotated += [orig_dim[0]/2, orig_dim[1]/2,0]

    return aunp_coordinates_rotated, aunp_coordinates_rotated_rotated

def update_translations_from_shifts(tilt_axes_angles, translations, shifts):
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    rotations = R.from_euler('Z', tilt_axes_angles, degrees=True)
    shifts_rotated = rotations.inv().apply(np.array([[a[0],a[1],0.0] for a in shifts]))
    translations += shifts_rotated[:,[0,1]]
    return translations
    