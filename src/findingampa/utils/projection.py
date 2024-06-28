import torch
from pathlib import Path
from findingampa.project_management import FAContext

def load_aunps(aunp_filename: Path, device):
    import numpy as np
    aunps = np.genfromtxt(aunp_filename,usecols=(2,3,4))
    aunps = aunps * 4.0
    ones = np.ones((aunps.shape[0], 1))
    aunps = np.hstack((aunps, ones)) 
    aunps = torch.tensor(aunps,device=device)
    return aunps

def project_aunp_model_mine(aunps, matrix, im_shape, r, device):
    projection = torch.mm(matrix, aunps.T).T[:,[0,1]]
    model_image = torch.ones(im_shape[0], im_shape[1], device=device)

    y = torch.arange(0, model_image.shape[0], device=device)
    x = torch.arange(0, model_image.shape[1], device=device)
    grid_y, grid_x = torch.meshgrid(y, x)
    for i in range(projection.shape[0]):
        temp_y = grid_y - projection[i, 1]
        temp_x = grid_x - projection[i, 0]
        # Create the mask in-place
        mask = temp_x.pow_(2).add_(temp_y.pow_(2)) <= r*r

        # Apply the mask
        model_image[mask] -= 1
    return model_image

def project_aunp_model_libtilt(aunps, matrix, im_shape, r, device):
    from libtilt.atomic_models import coordinates_to_image_2d
    import torch.nn.functional as F
    from libtilt.shift.shift_image import shift_2d
    coordinate_projection = torch.mm(matrix, aunps.T).T[:,[0,1]]
    projection = coordinates_to_image_2d(coordinates=coordinate_projection, image_shape=im_shape, yx=False)
    y = torch.arange(0, 20, device=device)
    x = torch.arange(0, 20, device=device)
    grid_y, grid_x = torch.meshgrid(y, x)
    temp_y = grid_y - 9.5
    temp_x = grid_x - 9.5
    # Create the mask in-place
    circle = (temp_x.pow_(2).add_(temp_y.pow_(2)) <= r*r) * 1.0
    #circle = shift_2d(circle, shifts=torch.tensor([-10, -10],device=device))
    #plt.imshow(circle.cpu()[-20:,-20:])
    del grid_x, grid_y, temp_x, temp_y
    torch.cuda.synchronize()
    projection = F.conv2d(projection[None,None],circle[None,None],padding='same')
    return projection[0,0], circle

def aunp_to_image2d(coordinate_projection, im_shape, r, device):
    from libtilt.atomic_models import coordinates_to_image_2d
    import torch.nn.functional as F
    coordinate_projection = torch.tensor(coordinate_projection,device=device)
    projection = coordinates_to_image_2d(coordinates=coordinate_projection, image_shape=im_shape, yx=False)
    y = torch.arange(0, 20, device=device)
    x = torch.arange(0, 20, device=device)
    grid_y, grid_x = torch.meshgrid(y, x)
    temp_y = grid_y - 9.5
    temp_x = grid_x - 9.5
    # Create the mask in-place
    circle = (temp_x.pow_(2).add_(temp_y.pow_(2)) <= r*r) * 1.0
    #circle = shift_2d(circle, shifts=torch.tensor([-10, -10],device=device))
    #plt.imshow(circle.cpu()[-20:,-20:])
    del grid_x, grid_y, temp_x, temp_y
    torch.cuda.synchronize()
    projection = F.conv2d(projection[None,None],circle[None,None],padding='same')
    return projection[0,0], circle

def compare_mine_and_libtilt(
        ctx: FAContext,
        tomogram_name: str,
        tilt: int,
        aunp_filename: Path
):
    import matplotlib.pyplot as plt
    import mrcfile
    import starfile
    from findingampa.utils.transforms import load_relion_matrices
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fig, axs = plt.subplots(1, 2, figsize=(10, 5)) # Create a figure and a set of subplots

    aunps = load_aunps(aunp_filename,device)
    tomogram_infos = starfile.read(ctx.obj.project.settings.FARelionTomoStar)

    tomogram_info = tomogram_infos["global"][tomogram_infos["global"]["rlnTomoName"] == tomogram_name].iloc[0]
    tilt_info = tomogram_infos[tomogram_name]
    tilt_series = torch.as_tensor(mrcfile.read(tomogram_info["rlnTomoTiltSeriesName"]),device=device)
    matrices = load_relion_matrices(tilt_info,device)

    mine = project_aunp_model_mine(aunps, matrices[tilt], tilt_series.shape[1:], 8.3, device)

    axs[0].imshow(mine.cpu().numpy())

    libtilt = project_aunp_model_libtilt(aunps, matrices[tilt], tilt_series.shape[1:], 8.3, device)
    axs[1].imshow(libtilt.cpu().numpy())
    plt.show()

