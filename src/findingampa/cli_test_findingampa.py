import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 
import typer
import starfile
from pathlib import Path
from types import SimpleNamespace
import pandas as pd

from findingampa.project_management import Project, FAContext, FAGlobal, Tomogram, RelionTomogram
from typing import Optional
from rich import print

app = typer.Typer(
)

def version_callback(value: bool):
    from findingampa import __version__ 
    if value:
        typer.echo(f"Finding AMPA Version: {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    ctx: FAContext,
    project: Path = typer.Option(None, help="Path to wanted project file",rich_help_panel="Expert Options"),
    imod_path: Path = typer.Option("", help="Path to IMOD binaries"),
    version: bool = typer.Option(None, "--version", callback=version_callback, is_eager=True),
):
    """Finding AMPA processing pipeline"""
    if project is None:
        project = Path.cwd() / "finding_ampa" / "finding_ampa_project.star"
        if not project.exists():
            ctx.obj = SimpleNamespace(project = None, imod_path = imod_path, project_path = None), 
            return
    project_object = starfile.read(project,always_dict=True)
    project_object["data"] = project_object["data"].apply(lambda x: x.to_dict(),axis=1).to_list()
    project_object = Project(**project_object)
    relion_tomograms = starfile.read(project_object.settings.FARelionTomoStar)["global"]
    relion_tomograms = {x["rlnTomoName"]: RelionTomogram(**x) for i, x in relion_tomograms.iterrows()}
    ctx.obj = FAGlobal(project = project_object, tomograms = project_object.data, project_path = project, project_directory = Path.cwd() / "finding_ampa", relion_tomograms = relion_tomograms)

@app.command()
def test_imod_backprojection(
    x_offset:float,
    y_offset:float,
    z_offset:float,
):
    import imodmodel
    import mrcfile
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    from findingampa.utils.imod import read_transforms_from_tomo_to_tilt
    from findingampa.utils.transforms import aunp_coordinates_to_tiltseries
    
    try:
        tomo_name = list(Path.cwd().glob("*.rawtlt"))[0].stem
    except IndexError:
        typer.echo("No rawtlt file found")
        raise typer.Exit()
    
    aunp_coordinates = starfile.read("aunps/aunp_tm_BP_active_zone_0.star")
    aunp_coordinates = aunp_coordinates[['faCoordinateX', 'faCoordinateY', 'faCoordinateZ']].to_numpy()
    tilt_angle, rotation_angle, translations = read_transforms_from_tomo_to_tilt(f"{tomo_name}.tlt", f"{tomo_name}.xf")
    tilt_angle += 25.8
    print(tilt_angle)
    print(rotation_angle)
    print(translations)
    with mrcfile.open(f"{tomo_name}_full_rec_BP_flat.mrc") as mrc:
        tomo_dim = mrc.header["nx"], mrc.header["ny"], mrc.header["nz"]
    with mrcfile.open(f"{tomo_name}_ali.mrc") as mrc:
        ali_dim = mrc.header["nx"], mrc.header["ny"]
    with mrcfile.open(f"{tomo_name}.mrc") as mrc:
        ori_dim = mrc.header["nx"], mrc.header["ny"]
    
    ali_coordinates, tilt_series_coordinates = aunp_coordinates_to_tiltseries(aunp_coordinates, tilt_angle, rotation_angle, translations, tomo_dim, ali_dim, ori_dim, x_offset=x_offset, y_offset=x_offset, z_offset=z_offset)
    with open("test_coordinates_rotated_rotated.txt", "w") as f:
        for i, aunps in enumerate(tilt_series_coordinates):
            for j, coordinates in enumerate(aunps):
                f.write(f"1 {i+1} {coordinates[0]} {coordinates[1]} {j}\n")
    with open("test_coordinates_rotated.txt", "w") as f:
        for i, aunps in enumerate(ali_coordinates):
            for j, coordinates in enumerate(aunps):
                f.write(f"1 {i+1} {coordinates[0]} {coordinates[1]} {j}\n")
    

    tilt_coordinates = imodmodel.read("aunps_tiltseries.mod")
    ali_ref = imodmodel.read("aunps_ali.mod")
    print(ali_coordinates.reshape(-1,3)[:,[0,1]] - ali_ref[['x', 'y']].to_numpy())
    print(tilt_series_coordinates.reshape(-1,3)[:,[0,1]] - tilt_coordinates[['x', 'y']].to_numpy())

@app.command()
def test_aunp_cc(radius:float=8.0,plot:bool=True):
    import mrcfile
    import torch
    import matplotlib.pyplot as plt
    import imodmodel
    import numpy as np
    from findingampa.utils.projection import aunp_to_image2d

    from findingampa.utils.imod import read_transforms_from_tomo_to_tilt
    from findingampa.utils.transforms import aunp_coordinates_to_tiltseries, update_translations_from_shifts

    from torch_fourier_filter.whitening import whitening_filter
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    try:
        tomo_name = list(Path.cwd().glob("*.rawtlt"))[0].stem
    except IndexError:
        typer.echo("No rawtlt file found")
        raise typer.Exit()
    debug_path = Path("debug/").mkdir(exist_ok=True)
    aunp_coordinates = starfile.read("aunps/aunp_tm_BP_active_zone_all.star")
    aunp_coordinates = aunp_coordinates[np.logical_or(aunp_coordinates["postsynapse"]=="True",aunp_coordinates["active_zone"]==0)]
    aunp_coordinates_orig = aunp_coordinates[['faCoordinateX', 'faCoordinateY', 'faCoordinateZ']].to_numpy()
    tilt_angle, rotation_angle, translations = read_transforms_from_tomo_to_tilt(f"{tomo_name}.tlt", f"{tomo_name}.xf")
    tilt_angle += 25.8
    with mrcfile.open(f"{tomo_name}_full_rec_BP_flat.mrc") as mrc:
        tomo_dim = mrc.header["nx"], mrc.header["ny"], mrc.header["nz"]
    with mrcfile.open(f"{tomo_name}_ali.mrc") as mrc:
        ali_dim = mrc.header["nx"], mrc.header["ny"]
    with mrcfile.open(f"{tomo_name}.mrc") as mrc:
        ori_dim = mrc.header["nx"], mrc.header["ny"]
    
    ali_coordinates, tilt_series_coordinates = aunp_coordinates_to_tiltseries(aunp_coordinates_orig, tilt_angle, rotation_angle, translations, tomo_dim, ali_dim, ori_dim)
    with mrcfile.open(f"{tomo_name}.mrc") as mrc:
        tilt_series = torch.as_tensor(mrc.data.copy(),device=device)

    aunp_coordinates = torch.as_tensor(tilt_series_coordinates,device=device)
    #tilt_series = torch.as_tensor(tilt_series,device=device)

    if plot:
        # Arrange len(matrices) subplots in a
        fig, axs = plt.subplots(2, len(tilt_series), figsize=(20, 4)) 
    shifts = []
    dmrc = []
    for i in range(tilt_series.shape[0]):
        model_image, C = aunp_to_image2d(tilt_series_coordinates[:,i,[0,1]], tilt_series.shape[1:], radius, device)
        if plot:
            dmrc.append(model_image.cpu().numpy())
        model_image *= -1.0
        dft = torch.fft.rfft2(tilt_series[i])
        wf = whitening_filter(dft, tilt_series[i].shape, tilt_series[i].shape, rfft=True )
        image_product = wf * dft * (wf * torch.fft.rfft2(model_image)).conj()
        cc_image = torch.fft.fftshift(torch.fft.irfft2(image_product))
        
        sub_orig = cc_image.real[cc_image.real.shape[0]//2-100:cc_image.real.shape[0]//2+100,cc_image.real.shape[1]//2-100:cc_image.real.shape[1]//2+100]
        

        import lmfit
        from lmfit.lineshapes import gaussian2d, lorentzian
        model = lmfit.models.Gaussian2dModel()
        z = sub_orig.cpu().detach().numpy().flatten()
        z = z - np.mean(z)
        z = z/np.std(z)
        # create x and y array for z values
        x = np.arange(sub_orig.shape[0])
        y = np.arange(sub_orig.shape[1])
        x, y = np.meshgrid(x, y)
        x = x.flatten()
        y = y.flatten()
        params = model.guess(z, x, y)
        params['centerx'].set(value=100)
        params['centery'].set(value=100)
        params['sigmax'].set(value=3,max=40)
        params['sigmay'].set(value=3,max=40)
        result = model.fit(z, x=x, y=y, params=params)
        #lmfit.report_fit(result)
        
        peak_pos = (sub_orig==torch.max(sub_orig)).nonzero()
        shift = (peak_pos-torch.tensor([100,100],device=device))[0]
        print(shift)
        #shifts.append(shift.cpu().detach().numpy())
        shift_parab = (result.best_values["centerx"]-100.5,result.best_values["centery"]-100.5)
        shifts.append(shift_parab)
        if plot:
            axs[0, i].imshow(sub_orig.cpu().detach())
            # disable axes
            axs[0, i].axis('off')
            axs[1, i].axis('off')
            fig.tight_layout()
            plt.savefig("test.png")
    if plot:
        with mrcfile.new(f"debug/model.mrc",overwrite=True) as mrc:
            mrc.set_data(np.stack(dmrc))
    print("Updated translations")
    translations = update_translations_from_shifts(rotation_angle, translations, np.array(shifts)[:,:])
    ali_coordinates, tilt_series_coordinates = aunp_coordinates_to_tiltseries(aunp_coordinates_orig, tilt_angle, rotation_angle, translations, tomo_dim, ali_dim, ori_dim)

    aunp_coordinates = torch.as_tensor(tilt_series_coordinates,device=device)
    #tilt_series = torch.as_tensor(tilt_series,device=device)

   
    for i in range(tilt_series.shape[0]):
        model_image, C = aunp_to_image2d(tilt_series_coordinates[:,i,[0,1]], tilt_series.shape[1:], radius, device)
        model_image *= -1.0
        image_product = torch.fft.fft2(tilt_series[i]) * torch.fft.fft2(model_image).conj()
        cc_image = torch.fft.fftshift(torch.fft.ifft2(image_product))
        
        sub_orig = cc_image.real[cc_image.real.shape[0]//2-100:cc_image.real.shape[0]//2+100,cc_image.real.shape[1]//2-100:cc_image.real.shape[1]//2+100]
        

        import lmfit
        from lmfit.lineshapes import gaussian2d, lorentzian
        model = lmfit.models.Gaussian2dModel()
        z = sub_orig.cpu().detach().numpy().flatten()
        z = z - np.mean(z)
        z = z/np.std(z)
        # create x and y array for z values
        x = np.arange(sub_orig.shape[0])
        y = np.arange(sub_orig.shape[1])
        x, y = np.meshgrid(x, y)
        x = x.flatten()
        y = y.flatten()
        params = model.guess(z, x, y)
        params['centerx'].set(value=100)
        params['centery'].set(value=100)
        params['sigmax'].set(value=3,max=40)
        params['sigmay'].set(value=3,max=40)
        result = model.fit(z, x=x, y=y, params=params)
        #lmfit.report_fit(result)
        
        peak_pos = (sub_orig==torch.max(sub_orig)).nonzero()
        shift = (peak_pos-torch.tensor([100,100],device=device))[0]
        print(shift)
        shift_parab = (result.best_values["centerx"]-100.5,result.best_values["centery"]-100.5)
        print(shift_parab)

        with open("optimized.xf","w") as f:
            for i in range(len(rotation_angle)):
                f.write(f"{np.cos(np.deg2rad(rotation_angle[i]))} {-np.sin(np.deg2rad(rotation_angle[i]))} {np.sin(np.deg2rad(rotation_angle[i]))} {np.cos(np.deg2rad(rotation_angle[i]))} {translations[i,0]} {translations[i,1]}\n")


@app.command()
def test_aunp_cc_tarotation(radius:float=8.0,plot:bool=True):
    import mrcfile
    import torch
    import matplotlib.pyplot as plt
    import imodmodel
    import numpy as np
    from findingampa.utils.projection import aunp_to_image2d

    from findingampa.utils.imod import read_transforms_from_tomo_to_tilt
    from findingampa.utils.transforms import aunp_coordinates_to_tiltseries, update_translations_from_shifts

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    try:
        tomo_name = list(Path.cwd().glob("*.rawtlt"))[0].stem
    except IndexError:
        typer.echo("No rawtlt file found")
        raise typer.Exit()
    
    aunp_coordinates = imodmodel.read("aunps/combined_savedpicks.mod")
    aunp_coordinates_orig = aunp_coordinates[['x', 'y', 'z']].to_numpy()
    tilt_angle, rotation_angle, translations = read_transforms_from_tomo_to_tilt(f"{tomo_name}.tlt", f"{tomo_name}.xf")
   
    with mrcfile.open(f"{tomo_name}_full_rec_BP.mrc") as mrc:
        tomo_dim = mrc.header["nx"], mrc.header["ny"], mrc.header["nz"]
    with mrcfile.open(f"{tomo_name}_ali.mrc") as mrc:
        ali_dim = mrc.header["nx"], mrc.header["ny"]
    with mrcfile.open(f"{tomo_name}.mrc") as mrc:
        ori_dim = mrc.header["nx"], mrc.header["ny"]
    
    
    with mrcfile.open(f"{tomo_name}.mrc") as mrc:
        tilt_series = torch.as_tensor(mrc.data.copy(),device=device)

    
    #tilt_series = torch.as_tensor(tilt_series,device=device)

    if plot:
        # Arrange len(matrices) subplots in a
        fig, axs = plt.subplots(11, len(tilt_series), figsize=(20, 4)) 
    
    shifts = []
    for j, ta_o in enumerate(np.arange(-0.5,0.51,0.1)):
        for i in range(tilt_series.shape[0]):
            ali_coordinates, tilt_series_coordinates = aunp_coordinates_to_tiltseries(aunp_coordinates_orig, tilt_angle, rotation_angle+ta_o, translations, tomo_dim, ali_dim, ori_dim)
            aunp_coordinates = torch.as_tensor(tilt_series_coordinates,device=device)
            model_image, C = aunp_to_image2d(tilt_series_coordinates[:,i,[0,1]], tilt_series.shape[1:], radius, device)
            model_image *= -1.0
            image_product = torch.fft.fft2(tilt_series[i]) * torch.fft.fft2(model_image).conj()
            cc_image = torch.fft.fftshift(torch.fft.ifft2(image_product))
            
            sub_orig = cc_image.real[cc_image.real.shape[0]//2-100:cc_image.real.shape[0]//2+100,cc_image.real.shape[1]//2-100:cc_image.real.shape[1]//2+100]
            

            import lmfit
            from lmfit.lineshapes import gaussian2d, lorentzian
            model = lmfit.models.Gaussian2dModel()
            z = sub_orig.cpu().detach().numpy().flatten()
            z = z - np.mean(z)
            z = z/np.std(z)
            # create x and y array for z values
            x = np.arange(sub_orig.shape[0])
            y = np.arange(sub_orig.shape[1])
            x, y = np.meshgrid(x, y)
            x = x.flatten()
            y = y.flatten()
            params = model.guess(z, x, y)
            params['centerx'].set(value=100)
            params['centery'].set(value=100)
            params['sigmax'].set(value=3,max=40)
            params['sigmay'].set(value=3,max=40)
            result = model.fit(z, x=x, y=y, params=params)
            lmfit.report_fit(result)
            
            peak_pos = (sub_orig==torch.max(sub_orig)).nonzero()
            shift = (peak_pos-torch.tensor([100,100],device=device))[0]
            #print(shift)
            #shifts.append(shift.cpu().detach().numpy())
            shift_parab = (result.best_values["centerx"]-100.5,result.best_values["centery"]-100.5)
            shifts.append(shift_parab)
            if plot:
                axs[j, i].imshow(sub_orig.cpu().detach())
                axs[j,i].set_title(f"{result.best_values['amplitude']:.2f} ")
                # disable axes
                axs[j, i].axis('off')
    fig.tight_layout()
    plt.show()

from enum import Enum

class Tomotype(str, Enum):
    BP = "BP"
    DDW = "DDW"

@app.command()
def select_aunp_picks(
    tomotype: Tomotype = Tomotype.BP,
    high_pass: Optional[float] = None,
    threshold: float = 3.8
):
    import torch
    import mrcfile
    import numpy as np
    from imodmodel import ImodModel
    from imodmodel.models import Object
    from findingampa.utils.analysis import select_aunps
    try:
        tomo_name = list(Path.cwd().glob("*.rawtlt"))[0].stem
    except IndexError:
        typer.echo("No rawtlt file found")
        raise typer.Exit()

    azs = Path("active_zonograms").glob("active_zonogram_*.mrc")
    azs = list(azs)
    print(f"Selecting AuNPS in {len(azs)} active zones")
    az_picks = []
    postsyn_picks = []
    for az in azs:
        with mrcfile.open(az, permissive=True) as mrc:
            res_ddw = torch.tensor(mrc.data.copy())
        az_data = np.load(az.with_suffix(".npy") ,allow_pickle=True)
        az_id = str(az.with_suffix("")).split("_")[-1]
        aunpfilename_filename = Path("aunps") / f"aunp_tm_{tomotype.value}{'_HP' + str(high_pass) if high_pass is not None else ''}.star"
        aunp_data = starfile.read(aunpfilename_filename)
        fig, significant_picks_mask, selected_aunp_pos, selected_aunp_pos_mask, selected_aunp_pos_postsyn_mask = select_aunps(aunp_data, az_data, res_ddw, threshold)
        diagnost_figure_filename = str(az.with_suffix(""))+"_selected_aunps.png"
        fig.savefig(diagnost_figure_filename)
        print(f"Saved diagnostic image {diagnost_figure_filename}")
        az_picks.append(selected_aunp_pos_mask)
        postsyn_picks.append(selected_aunp_pos_postsyn_mask)

    selected_aunp_pos = aunp_data.loc[significant_picks_mask, ["faTM_PeakX", "faTM_PeakY", "faTM_PeakZ"]].to_numpy()
    data_structure = pd.DataFrame(selected_aunp_pos, columns=["faCoordinateX", "faCoordinateY", "faCoordinateZ"])
    data_structure["active_zone"] = -1
    data_structure["postsynapse"] = False
    model = ImodModel()
    az_pick_filename = str(aunpfilename_filename.with_suffix(""))+f"_active_zone_noshift.mod"
    for i, az in enumerate(azs):
        data_structure.loc[az_picks[i], "active_zone"] = i
        az_id = str(az.with_suffix("")).split("_")[-1]
        filename = az_pick_filename[:-4] + f"_{az_id}.star"
        #starfile.write(data_structure.loc[az_picks[i]], filename)
        print(f"Saved picks to {filename}")
        model.objects.append(Object.new_scattered_points(points=selected_aunp_pos[az_picks[i]]))
        model.objects[-1].header.red = 1.0 - i *0.3
        model.objects[-1].header.green = 0 + i*0.2
        model.objects[-1].header.blue = 0 + i*0.2
    for i, postsyn in enumerate(postsyn_picks):
        data_structure.loc[postsyn, "postsynapse"] = True
    
    model.objects.append(Object.new_scattered_points(points=selected_aunp_pos[np.logical_and(
        data_structure["active_zone"] == -1,
        data_structure["postsynapse"] == True
    )]))
    model.objects[-1].header.red = 0
    model.objects[-1].header.green = 1.0
    model.objects[-1].header.blue = 0
    
    
    
    
    
    model.objects.append(Object.new_scattered_points(points=selected_aunp_pos[np.logical_and(
        data_structure["active_zone"] == -1,
        data_structure["postsynapse"] == False
    )]))
    model.objects[-1].header.red = 0
    model.objects[-1].header.green = 0
    model.objects[-1].header.blue = 1.0

    model.to_file(az_pick_filename)
    #starfile.write(az_picks, az_pick_filename)
    print(f"Saved picks to {az_pick_filename}")
    filename = az_pick_filename[:-4] + f"_allnoshift.star"
    #starfile.write(data_structure, filename)
    print(f"Saved picks to {filename}")

@app.command()
def create_ampa_positions():
    import numpy as np
    from scipy.spatial import cKDTree as KDTree
    from scipy.spatial import distance
    from scipy.spatial.transform import Rotation as R
    from imodmodel import ImodModel
    from imodmodel.models import Object, ObjectHeader, Contour, ContourHeader, IMAT,SLAN

    try:
        tomo_name = list(Path.cwd().glob("*.rawtlt"))[0].stem
    except IndexError:
        typer.echo("No rawtlt file found")
        raise typer.Exit()

    aunp_filename = "aunps/aunp_tm_BP_active_zone_all.star"
    aunp_data = starfile.read(aunp_filename)


    post_synapse = []
    for ps_txt in Path("aunps/").glob("postsynapticmembranes_*.txt"):
        post_synapse.append(np.loadtxt(ps_txt))
    post_synapse = np.concatenate(post_synapse)

    coordinates = aunp_data[['faCoordinateX', 'faCoordinateY', 'faCoordinateZ']].to_numpy()
    tree = KDTree(coordinates)
    raw_pairs = tree.query_pairs(15)  # Find pairs with distance less than 150A

    # Sort pairs by difference of distance to optimal distance
    sorted_pairs = sorted(raw_pairs, key=lambda pair: abs(distance.euclidean(coordinates[pair[0]], coordinates[pair[1]])-8.5))
    pair_points = np.array([[pair[0],pair[1],*((coordinates[pair[0]]+coordinates[pair[1]])/2)] for pair in sorted_pairs])

    ps_tree = KDTree(post_synapse)
    distances, point_idx = ps_tree.query(pair_points[:,2:])
    pair_points = pair_points[distances < 20]
    print(f"{len(pair_points)} pairs found")
    points_idx = point_idx[distances < 20]
    model = ImodModel()
    model.objects.append(Object(
                header=ObjectHeader(contsize=len(pair_points)),
                contours=[
                    Contour(header=ContourHeader(
                        psize=4, 
                        flags=16, 
                        time=0, 
                        surf=0), 
                        points=np.array([coordinates[int(points[0])], coordinates[int(points[1])], points[2:], post_synapse[int(points_idx[i])]]),
                        extra=[]
                        ) for i, points in enumerate(pair_points)
                    ], 
                imat = IMAT()    
            ))
    v1s = coordinates[pair_points[:,0].astype(int)] - coordinates[pair_points[:,1].astype(int)]
    # X angle should 
    v2s = post_synapse[points_idx] - pair_points[:,2:]
    
    rots = [R.align_vectors([[1, 0, 0],[0,1,0]], [v1,v2])[0] for v1, v2 in zip(v1s, v2s)]
    eulers = [rot.as_euler("yxz", degrees=True) + [0,0,90] for rot in rots]
    print(eulers)
    
    model.slicer_angles = [SLAN(time=1,
                                angles=eulers[i],
                center=pair_points[i][2:],
                label="test") for i in range(len(pair_points))]
    model.to_file(f"test1.mod")

