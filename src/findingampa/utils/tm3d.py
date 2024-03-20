

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