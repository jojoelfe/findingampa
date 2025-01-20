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

def insert_into_image_2d(
    data: torch.Tensor,
    coordinates: torch.Tensor,
    image: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Insert values into a 2D image with trilinear interpolation (rasterisation).

    Parameters
    ----------
    data: torch.Tensor
        `(...)` array of values to be inserted into the DFT.
    coordinates: torch.Tensor
        `(..., 3)` array of 3D coordinates for each value in `data`.
    image: torch.Tensor
        `(d, d, d)` volume containing the discrete Fourier transform into which data will be inserted.
    weights: torch.Tensor
        `(d, d, d)` volume containing the weights associated with each voxel of `dft`

    Returns
    -------
    dft, weights: tuple[torch.Tensor, torch.Tensor]
        The dft and weights after updating with data from `slices` at `slice_coordinates`.
    """
    import einops
    from typing import Literal
    if data.shape != coordinates.shape[:-1]:
        raise ValueError('One coordinate triplet is required for each value in data.')

    # linearise data and coordinates
    data, _ = einops.pack([data], pattern='*')
    coordinates, _ = einops.pack([coordinates], pattern='* zyx')
    coordinates = coordinates.float()

    # only keep data and coordinates inside the volume
    in_volume_idx = (coordinates >= 0) & (coordinates <= torch.tensor(image.shape,device="cuda") - 1)
    in_volume_idx = torch.all(in_volume_idx, dim=-1)
    data, coordinates = data[in_volume_idx], coordinates[in_volume_idx]

    # calculate and cache floor and ceil of coordinates for each piece of slice data
    corner_coordinates = torch.empty(size=(data.shape[0], 2, 2), dtype=torch.long, device="cuda")
    corner_coordinates[:, 0] = torch.floor(coordinates)  # for lower corners
    corner_coordinates[:, 1] = torch.ceil(coordinates)  # for upper corners

    # cache linear interpolation weights for each data point being inserted
    _weights = torch.empty(size=(data.shape[0], 2, 2),device="cuda")  # (b, 2, yx)
    _weights[:, 1] = coordinates - corner_coordinates[:, 0]  # upper corner weights
    _weights[:, 0] = 1 - _weights[:, 1]  # lower corner weights

    def add_data_at_corner(y: Literal[0, 1], x: Literal[0, 1]):
        w = einops.reduce(_weights[:, [y, x], [0, 1]], 'b yx -> b', reduction='prod')
        yc, xc = einops.rearrange(corner_coordinates[:, [y, x], [0, 1]], 'b yx -> yx b')
        image.index_put_(indices=(yc, xc), values=w * data, accumulate=True)
        weights.index_put_(indices=(yc, xc), values=w, accumulate=True)

    add_data_at_corner(0, 0)
    add_data_at_corner(0, 1)
    add_data_at_corner(1, 0)
    add_data_at_corner(1, 1)
    return image, weights


def coordinates_to_image_2d(
    coordinates: torch.Tensor,
    image_shape: tuple[int, int],
    yx: bool = True,
) -> torch.Tensor:
    """Render an image from 2D coordinates by inserting ones with linear interpolation.

    Parameters
    ----------
    coordinates: torch.Tensor
        `(..., 2)` array of 3D coordinates at which ones will be inserted into
         the output image.
    image_shape: tuple[int, int]
        `(h, w)` shape of the output image.
    yx: bool
        Whether the coordinates are ordered yx (`True`) or xy (`False`).

    Returns
    -------
    image: torch.Tensor
        `(h, w)` output volume.
    """
    import einops
    image = torch.zeros(
        size=image_shape, dtype=torch.float32, device=coordinates.device
    )
    if yx is False:
        coordinates = torch.flip(coordinates, dims=(-1,))
    coordinates, ps = einops.pack([coordinates], pattern='* coords')
    b, _ = coordinates.shape
    ones = torch.ones(size=(b,), dtype=torch.float32, device=coordinates.device)
    weights = torch.zeros_like(image)
    image, weights = insert_into_image_2d(
        data=ones, coordinates=coordinates, image=image, weights=weights
    )
    return image


def aunp_to_image2d(coordinate_projection, im_shape, r, device):
    import torch.nn.functional as F
    coordinate_projection = torch.tensor(coordinate_projection,device=device)
    projection = coordinates_to_image_2d(coordinates=coordinate_projection, image_shape=im_shape, yx=False)
    y = torch.arange(0, 20, device=device)
    x = torch.arange(0, 20, device=device)
    grid_y, grid_x = torch.meshgrid(y, x)
    temp_y = grid_y - 10
    temp_x = grid_x - 10
    # Create the mask in-place
    gaussian = torch.exp(-(temp_x.pow(2)+temp_y.pow(2))/(2*r*r))
    #circle = shift_2d(circle, shifts=torch.tensor([-10, -10],device=device))
    #plt.imshow(circle.cpu()[-20:,-20:])
    del grid_x, grid_y, temp_x, temp_y
    torch.cuda.synchronize()
    projection = F.conv2d(projection[None,None],gaussian[None,None],padding='same')
    return projection[0,0], gaussian

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

