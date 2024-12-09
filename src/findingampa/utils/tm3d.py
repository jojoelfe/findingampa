import numpy as np
import torch.nn.functional as F
import torch
import einops

import torch.nn as nn
import torch.optim as optim

def calc_3dtm(tomo, high_pass=None):
    from pytom_tm.tmjob import TMJob
    from pytom_tm.parallel import run_job_parallel
    from findingampa.utils.assets import FA_ASSETS
    from pathlib import Path
    from skimage.feature import peak_local_max


    tmfolder = Path('./') / "TM"
    tmfolder.mkdir(exist_ok=True)
    job = TMJob(
        "0",
        20,
        tomo,
        FA_ASSETS / "templates" / "aunp01" / "temaplte.mrc",
        FA_ASSETS / "templates" / "aunp01" / "mask.mrc",
        tmfolder,
        angle_increment=str(FA_ASSETS / "templates" / "aunp01" / "angles.txt"),
        mask_is_spherical=True,
        tilt_angles=[-48,48],
        tilt_weighting=None,
        search_x=None,
        search_y=None,
        search_z=None,
        tomogram_mask=None,
        voxel_size=10.0,
        low_pass=None,
        high_pass=high_pass,
        dose_accumulation=None,
        ctf_data=None,
        whiten_spectrum=True,
        rotational_symmetry=1,
        particle_diameter=None,
        random_phase_correction=False,
        rng_seed=5,
    )

    score_volume, angle_volume = run_job_parallel(
        job, (2,2,2), [0]
    )
    #mrcfile.write(f"/scratch/pompeii/elferich/gouaux_tomo/ProcessingJE/AuNP_TM/{tomo.name.removesuffix('.mrc')}_score.mrc", score_volume.T, overwrite=True, voxel_size=10.0)
    peaks = peak_local_max(score_volume, min_distance=2, threshold_abs=0.3, exclude_border=10)

    locations_tomo = torch.Tensor(peaks[:,::-1].copy()+ np.array([0,0,0])) # FRom XYZ to ZYX
    return locations_tomo, torch.Tensor(peaks).int(), score_volume

class Gaussian3DList(nn.Module):
    def __init__(self, num_gaussians,sign=1):
        super(Gaussian3DList, self).__init__()
        self.amplitude = nn.Parameter(torch.ones(num_gaussians)*sign)
        self.x0 = nn.Parameter(torch.zeros(num_gaussians))
        self.y0 = nn.Parameter(torch.zeros(num_gaussians))
        self.z0 = nn.Parameter(torch.zeros(num_gaussians))
        self.sigma_x = nn.Parameter(torch.ones(num_gaussians))
        self.sigma_y = nn.Parameter(torch.ones(num_gaussians))
        self.sigma_z = nn.Parameter(torch.ones(num_gaussians))

    def forward(self, x, y, z):
        
        x = x.unsqueeze(0)  # Shape: (num_points, num_gaussians)
        y = y.unsqueeze(0)
        z = z.unsqueeze(0)
        return self.amplitude.unsqueeze(1) * torch.exp(
            -((x - self.x0.unsqueeze(1)) ** 2 / (2 * self.sigma_x.unsqueeze(1) ** 2) +
              (y - self.y0.unsqueeze(1)) ** 2 / (2 * self.sigma_y.unsqueeze(1) ** 2) +
              (z - self.z0.unsqueeze(1)) ** 2 / (2 * self.sigma_z.unsqueeze(1) ** 2))
        )  # Sum over the gaussians

def fit_gaussian_3d_list(score_volume, locations, boxsize, sign=1):
    from torch_subpixel_crop import subpixel_crop_3d

    num_gaussians = len(locations)
    model = Gaussian3DList(num_gaussians,sign=sign).to(score_volume.device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Prepare the 4D tensor of boxes
    #boxes = []
    #for loc in locations:
    #    x, y, z = loc
    #    region = score_volume[x-boxsize//2:x+boxsize//2+1, y-boxsize//2:y+boxsize//2+1, z-boxsize//2:z+boxsize//2+1]
        
    #    boxes.append(region)
    boxes = subpixel_crop_3d(score_volume, locations, boxsize)
    #boxes = torch.stack(boxes).to(score_volume.device)  # Shape: (num_gaussians, boxsize, boxsize, boxsize)

    # Prepare the coordinates
    coords = torch.stack(torch.meshgrid(
        torch.arange(boxsize),
        torch.arange(boxsize),
        torch.arange(boxsize)
    ), dim=-1).reshape(-1, 3).to(score_volume.device) - torch.tensor([boxsize//2, boxsize//2, boxsize//2], device=score_volume.device)
    x_coords = coords[:, 0].float()
    y_coords = coords[:, 1].float()
    z_coords = coords[:, 2].float()

    for _ in range(1000):  # Number of iterations
        optimizer.zero_grad()
        output = model(x_coords, y_coords, z_coords).reshape(num_gaussians, boxsize, boxsize, boxsize)
        loss = criterion(output, boxes)
        loss.backward()
        optimizer.step()

    fitted_params = []
    for i in range(num_gaussians):
        fitted_params.append({
            'amplitude': model.amplitude[i].item(),
            'x0': model.x0[i].item(),
            'y0': model.y0[i].item(),
            'z0': model.z0[i].item(),
            'sigma_x': model.sigma_x[i].item(),
            'sigma_y': model.sigma_y[i].item(),
            'sigma_z': model.sigma_z[i].item(),
            'loss': loss.item()
        })

    return fitted_params



def test_3d_cc_fft(vol,r):
    import numpy as np
    import torch
    import torch.fft
    from libtilt.shift.shift_image import shift_3d
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    circle = torch.zeros(vol.shape[0],vol.shape[1],vol.shape[2],device=device)
    y = torch.arange(0, vol.shape[1], device=device)
    x = torch.arange(0, vol.shape[2], device=device)
    z = torch.arange(0, vol.shape[0], device=device)
    grid_z, grid_y, grid_x = torch.meshgrid(z, y, x)
    temp_y = grid_y - 9.5
    temp_x = grid_x - 9.5
    temp_z = grid_z - 9.5
    # Create the mask in-place
    circle = (temp_x.pow_(2).add_(temp_y.pow_(2)).add_(temp_z.pow_(2)) <= r*r) * -1.0
    circle = shift_3d(circle, shifts=torch.tensor([-10, -10, -10],device=device))
    #plt.imshow(circle.cpu()[-20:,-20:])
    del grid_x, grid_y, temp_x, temp_y, grid_z, temp_z
    
    image_product = torch.fft.fftn(vol) * torch.fft.fftn(circle).conj()
    cc_image = torch.fft.ifftn(image_product).real
    
    return cc_image