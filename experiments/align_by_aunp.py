import mrcfile
import starfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.fft
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def plot_cc(im, matrix, aunps, tilt=0, r=6,show_ccpeak=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    projection = torch.mm(matrix, aunps.T).T[:,[0,1]]
    model_image = torch.ones(im.shape[0], im.shape[1], device=device)
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

    #plt.imshow(model_image.cpu())
    #plt.show()
    #plt.imshow(im.cpu())
    #plt.show()
    image_product = torch.fft.fft2(im) * torch.fft.fft2(model_image).conj()
    cc_image = torch.fft.fftshift(torch.fft.ifft2(image_product))
    sub = cc_image.real[cc_image.real.shape[0]//2-100:cc_image.real.shape[0]//2+100,cc_image.real.shape[1]//2-100:cc_image.real.shape[1]//2+100]
    if show_ccpeak:
        show_ccpeak.imshow(sub.cpu())
    #plt.imshow(cc_image.real.cpu())
    #plt.show()
    # Where is the peak?
    #from skimage.registration import phase_cross_correlation
    #shift, error, diffphase = phase_cross_correlation(im.cpu().numpy(), model_image.cpu().numpy(),normalization=None)
    #print(shift)
    #print(error)
    #print(diffphase)

    # find the peak in sub
    peak_pos = (sub==torch.max(sub)).nonzero()
    #print(peak_pos)
    shift_mine = peak_pos-torch.tensor([100,100],device=device)
    #print(shift_mine)
    # Simple crosscorrelation
    #cc_value = model_image * im
    #print(cc_value.sum())
    #print(torch.max(sub))
    return shift_mine.cpu().numpy()[0]
    #return torch.max(sub)-torch.median(sub)

def matrix_from_angles(loc,ya,za):
    from libtilt.transformations import Rx,Rz,T,Ry
    t=T(loc)
    ry=Ry(ya,zyx=False)
    rz=Rz(za,zyx=False)
    return t@rz@ry

def grid_search_angles():
    fig,ax=plt.subplots(5,5,figsize=(10,5))
    for y in range(-2,3):
        for z in range(-2,3):
            m = matrix_from_angles(loc,ya+y*0.5,za+z*0.5)
            plot_cc(im_data[5].to(device).detach(),m[0].type(torch.float64).to(device),aunps,r=8.3,show_ccpeak=ax[y+2,z+2])
    fig,ax=plt.subplots(5,5,figsize=(10,5))
    for y in range(-2,3):
        for z in range(-2,3):
            m = matrices[5]
            mn = m @ Rz(z*0.5).type(torch.float64)[0].to(device) @ Ry(y*0.5)[0].type(torch.float64).to(device)
            print(m)
            plot_cc(im_data[5].to(device).detach(),mn,aunps,r=8.3,show_ccpeak=ax[y+2,z+2])
        
        

def reine_angles_bycc(im, matrix, aunps):
    from libtilt.shift.shift_image import shift_2d
    from libtilt.rescaling.rescale_fourier import rescale_2d
    r_im, _ = rescale_2d(im, source_spacing=1.0, target_spacing=4.0, maintain_center=True)
    r=8.3 / 4.0
    r_aunps = aunps / 4.0
    circle = torch.zeros(r_im.shape[0],r_im.shape[1],device=device)
    y = torch.arange(0, r_im.shape[0], device=device)
    x = torch.arange(0, r_im.shape[1], device=device)
    grid_y, grid_x = torch.meshgrid(y, x)
    temp_y = grid_y - 9.5
    temp_x = grid_x - 9.5
    # Create the mask in-place
    circle = (temp_x.pow_(2).add_(temp_y.pow_(2)) <= r*r) * 1.0
    circle = shift_2d(circle, shifts=torch.tensor([-10, -10],device=device))
    #plt.imshow(circle.cpu()[-20:,-20:])
    del grid_x, grid_y, temp_x, temp_y
    torch.cuda.synchronize()

    to_optimize = matrix.clone().detach().requires_grad_(True)
    project_matrix_optimizer = torch.optim.Adam(
        params= [to_optimize,],
        lr=0.0005
    )

    proj_image = torch.zeros(r_im.shape[0], r_im.shape[1], device=device,requires_grad=True)
    for j in range(500):
        proj_image = torch.zeros(r_im.shape[0], r_im.shape[1], device=device,requires_grad=True)
        project_matrix_optimizer.zero_grad()
        projection = torch.mm(to_optimize, r_aunps.T).T[:,[0,1]]
        
        
        
        for i in range(projection.shape[0]):
            inv_co = torch.flip(projection[i], [0])
            proj_image = proj_image - shift_2d(circle, shifts=inv_co)
        image_product = torch.fft.fft2(r_im) * torch.fft.fft2(proj_image).conj()
        cc_image = torch.fft.fftshift(torch.fft.ifft2(image_product))
        sub= cc_image.real[cc_image.real.shape[0]//2-25:cc_image.real.shape[0]//2+26,cc_image.real.shape[1]//2-25:cc_image.real.shape[1]//2+26]
        ph= torch.log_softmax(sub,dim=0).min()
        loss = ph
        
        #cc_value = (proj_image * im).sum()
        loss.backward()
        project_matrix_optimizer.step()
        if j % 50 == 0:
            print(j, loss.item())
            #plt.imshow(sub.cpu().detach())
            #plt.show()
    return to_optimize

starfile_path = Path(('/scratch_pompeii/elferich/gouaux_tomo/ProcessingJE/manuscript01/relion_reconstruction/ImportTomo/job002/tomograms.star'))
data = starfile.read(starfile_path)
matrices = data['20231017_EGmilled24-2_60']
matrices = matrices.iloc[:,[0,1,2,3]].to_numpy()
matrices = [[s.strip('[]').split(',') for s in matrix] for matrix in matrices]
matrices = np.array([np.vstack([[float(x) for x in row] for row in matrix]) for matrix in matrices])
matrices = torch.tensor(matrices,device=device,requires_grad=True)

mrcfile_path = Path('/scratch/pompeii/elferich/gouaux_tomo/ProcessingJE/manuscript01/relion_reconstruction/culled_ts/20231017_EGmilled24-2_60_culled.mrc')
im_data = mrcfile.read(mrcfile_path)
im_data = torch.tensor(im_data,device='cpu')

aunps = np.genfromtxt('/scratch_pompeii/elferich/gouaux_tomo/ProcessingJE/manuscript01/relion_reconstruction/finding_ampa/pipeline_data/pipeline1/visualization/20231017_EGmilled24-2_60/combined_aunp.txt',delimiter=' ',usecols=(2,3,4))
aunps = aunps * 4.0
ones = np.ones((aunps.shape[0], 1))
aunps = np.hstack((aunps, ones)) 
aunps = torch.tensor(aunps,device=device)



if __name__ == '__main__':
    centered_matrices = []
    for tilt in range(len(matrices)):
        fig,axs = plt.subplots(1,3,figsize=(10,5))
        shift = plot_cc(im_data[tilt].to(device).detach(),matrices[tilt],aunps,r=8.3,show_ccpeak=axs[0])
        print(shift)
        new =  matrices[tilt].clone().detach()
        new[0,3] += shift[1]
        new[1,3] += shift[0]
        shift = plot_cc(im_data[tilt].to(device).detach(),new,aunps,r=8.3,show_ccpeak=axs[1])
        print(shift)
        centered_matrices.append(new.cpu().numpy())

        #ref = reine_angles_bycc(im_data[tilt].to(device).detach(),new,aunps)
        #shift = plot_cc(im_data[tilt].to(device).detach(),ref,aunps,r=8.3,show_ccpeak=axs[2])
        print(shift)
        plt.show()
        


    print(data['20231017_EGmilled24-2_60'])
    for tilt in range(len(matrices)):
        for i in range(4):
            arr_str = np.array2string(centered_matrices[tilt][i], precision=12, separator=',', suppress_small=True,max_line_width=1000)
            arr_str = arr_str.replace(' ', '')
            print(arr_str)
            data['20231017_EGmilled24-2_60'].iloc[tilt,i] = arr_str
    print(data['20231017_EGmilled24-2_60'])        
    starfile.write(data,starfile_path.parent / 'centered.star')