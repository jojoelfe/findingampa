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