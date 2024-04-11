@app.command()
def napari_animation():
    try:
        tomo_name = list(Path.cwd().glob("*.rawtlt"))[0].stem
    except IndexError:
        typer.echo("No rawtlt file found")
        raise typer.Exit()
    tomo = f"{tomo_name}_full_rec_BP_flat.mrc"

    import napari

@app.command()
def test_centerline(
    contour_file: Path = typer.Argument(...,help="Path to contour file")
):
    import imodmodel
    from shapely.geometry import Polygon
    from centerline.geometry import Centerline
    import matplotlib.pyplot as plt
    data = imodmodel.read(contour_file)
    # Split data by contour_id and iterate over each contour
    for contour_id, contour_data in data.groupby("contour_id"):
        if contour_id < 17630:
            continue
        plt.figure()
        # Get the contour coordinates
        contour = contour_data[["x","y"]].to_numpy()
        # Get the centerline coordinates
        polygon = Polygon(contour)
        attributes = {"id": 1, "name": "polygon", "valid": True}
        centerline = Centerline(polygon, interpolation_distance=2.0, **attributes)
        for a in centerline.geometry.geoms:
            for b in a.coords:
                print(b)
        plt.plot(*polygon.exterior.xy)
        plt.show()

@app.command()
def test_centerline2(
    contour_file: Path = typer.Argument(...,help="Path to contour file"),
    qpix_data_filename: Path = typer.Argument(...,help="Path to qpix data file"),
    the_paramter: float = typer.Argument(0.5,help="The parameter")
):
    import imodmodel
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import linemerge
    from centerline.geometry import Centerline
    import matplotlib.pyplot as plt
    data = imodmodel.read(contour_file)
    import mrcfile

    with mrcfile.open(qpix_data_filename) as mrc:
        qpixdata = mrc.data.copy()
    qpixdata = qpixdata - qpixdata.min()
    new_contour_data = []
    # Split data by contour_id and iterate over each contour
    cont_id = 1
    data["z"] = data["z"].round(0).astype(int)
    #previous_geoms = None
    surface_id = 1
    previous_surfaces = []
    for z, z_data in data.groupby("z"):

        #plt.figure()
        # Get the contour coordinates
        polygon_list = []
        for contour_id, contour_data in z_data.groupby("contour_id"):
            contour = contour_data[["x","y"]].to_numpy()
            if len(contour) < 3:
                continue
            polygon = Polygon(contour)
            polygon_list.append(Polygon(contour))
        
        # Get the centerline coordinates
        multipolygon = MultiPolygon(polygon_list)

        #for a in multipolygon.geoms:
        #    plt.plot(*a.exterior.xy)
        #plt.show()
        attributes = {"id": 1, "name": "polygon", "valid": True}
        centerline = Centerline(multipolygon, interpolation_distance=the_paramter, **attributes)
        # Only use centerline segments that are in the anotatedf membrane
        segments = [a for a in centerline.geometry.geoms if qpixdata[int(z),int(a.centroid.y),int(a.centroid.x)] > 0.5]
        # J
        cont = linemerge(segments)
        long_enough = [a for a in cont.geoms if a.length > 10.0]
        cont = linemerge(long_enough)
        current_surfaces = []
        for a in cont.geoms:
            current_surface = None
            prev_best = False
            for p_surface_id, other in previous_surfaces:
                intersection = a.intersection(other, grid_size=5.0)
                if intersection.length > 0.5 * min(a.length,other.length):
                    if prev_best:
                        if intersection.length < prev_best:
                            continue
                    print(f"Found a match {intersection.length} Contour {cont_id} Surface {p_surface_id}")
                    current_surface = p_surface_id
                    prev_best = intersection.length
            if current_surface is None:
                current_surface = surface_id
                surface_id += 1
            current_surfaces.append((current_surface,a))
            for b in a.coords:
                new_contour_data.append([current_surface, cont_id, b[0], b[1], z])
            cont_id += 1
        previous_surfaces = current_surfaces
    pd.DataFrame(new_contour_data).to_csv("temp_model.txt", sep=" ", header=None, index=False, mode="w")
            #plt.plot(*a.xy)


@app.command()
def test_skeletonize(
    qpix_data: Path = typer.Argument(...,help="Path to qpix data file")
):
    import mrcfile

    with mrcfile.open(qpix_data) as mrc:
        data = mrc.data.copy()
    data = data - data.min()
    print(data.min(),data.max())
    from skimage.morphology import skeletonize

    for z in range(data.shape[0]):
        skeleton = skeletonize(data[z,:,:])
        data[z,:,:] = skeleton
    with mrcfile.new(qpix_data.with_stem(qpix_data.stem + '_ske').with_suffix('.mrc'), overwrite=True) as mrc:
        mrc.set_data(data)

@app.command()
def test_membr(
    ctx: FAContext,
):
    from findingampa.pipeline1.pipeline1_steps import annotate_membrane
    #suffixes= ["_relion.mrc", "_ci.mrc","_cryocare.mrc","_mBP.mrc","_mSIRT.mrc"]
    suffixes= ["_mBP_rotx.mrc","_mSIRT_rotx.mrc"]
    for suf in suffixes:
        tomogram = ctx.obj.tomograms[0]
        tomogram.FAPL1TomoSuffixForMembrain = suf
        annotate_membrane(ctx,tomogram=tomogram, output_path=Path.cwd() / "finding_ampa" / f"pipeline_data_test{suf}" / "pipeline1")

@app.command()
def reconstruct(
    ctx: FAContext,
    tomogram: str = typer.Argument(...,help="Name of tomogram"),
    method: int = typer.Option(1,help="Reconstruction method"),
):
    from findingampa.utils.reconstruct import reconstruct_by_relion, reconstruct_by_libtilt
    if method == 1:
        output = reconstruct_by_relion(ctx.obj.project.settings.FARelionTomoStar,tomogram,binning=4.0)
        print(f"Reconstruction saved to {output}")
    elif method == 2:
        output = reconstruct_by_libtilt(ctx.obj.project.settings.FARelionTomoStar,tomogram,binning=4.0)
        print(f"Reconstruction saved to {output}")

@app.command()
def compare_projection(
    ctx: FAContext,
    tomogram: str = typer.Argument(...,help="Name of tomogram"),
    tilt: int = typer.Argument(0,help="Tilt angle"),
    aunp_filename: Path = typer.Argument(...,help="Path to aunp file")
):
    from findingampa.utils.projection import compare_mine_and_libtilt
    compare_mine_and_libtilt(ctx,tomogram,tilt,aunp_filename)

@app.command()
def test_3d_cc(
    tomogram: Path = typer.Argument(...,help="Path to tomogram"),
    r: float = typer.Option(2,help="Radius"),
):
    from findingampa.utils.tm3d import test_3d_cc_fft
    import torch
    import mrcfile
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    volume = torch.as_tensor(mrcfile.read(tomogram),device=device)
    cc_image = test_3d_cc_fft(volume,r)
    with mrcfile.new(tomogram.with_stem(tomogram.stem + '_cc').with_suffix('.mrc'), overwrite=True) as mrc:
        mrc.set_data(cc_image.cpu().numpy())

@app.command()
def test_dog(
    tomogram: Path = typer.Argument(...,help="Path to tomogram"),
    r1: float = typer.Option(2,help="Radius 1"),):
    pass

@app.command()
def refine_shifts_using_aunps(
    ctx: FAContext,
    tomogram: str = typer.Argument(...,help="Name of tomogram"),
    aunp_filename: Path = typer.Argument(...,help="Path to aunp file"),
    tomogram_starfile: Optional[Path] = typer.Option(None,help="Path to tomogram star file"),
    plot: bool = typer.Option(False,help="Plot the results")
):
    
    if tomogram_starfile is None:
        tomogram_starfile = ctx.obj.project.settings.FARelionTomoStar
    
    from findingampa.utils.transforms import load_relion_matrices
    from findingampa.utils.projection import load_aunps, project_aunp_model_libtilt
    import mrcfile
    import torch
    import starfile
    import matplotlib.pyplot as plt
    import numpy as np
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tomogram_infos = starfile.read(tomogram_starfile)

    tomogram_info = tomogram_infos["global"][tomogram_infos["global"]["rlnTomoName"] == tomogram].iloc[0]
    tilt_info = tomogram_infos[tomogram]

    matrices = load_relion_matrices(tilt_info, device)

    aunps = load_aunps(aunp_filename, device)
    tilt_series = torch.as_tensor(mrcfile.read(tomogram_info["rlnTomoTiltSeriesName"]),device=device)

    if plot:
        # Arrange len(matrices) subplots in a
        fig, axs = plt.subplots(2, len(matrices), figsize=(20, 4)) 

    for i in range(tilt_series.shape[0]):
        model_image, C = project_aunp_model_libtilt(aunps, matrices[i], tilt_series.shape[1:], 8.3, device)
        model_image *= -1.0
        image_product = torch.fft.fft2(tilt_series[i]) * torch.fft.fft2(model_image).conj()
        cc_image = torch.fft.fftshift(torch.fft.ifft2(image_product))
        sub_orig = cc_image.real[cc_image.real.shape[0]//2-100:cc_image.real.shape[0]//2+100,cc_image.real.shape[1]//2-100:cc_image.real.shape[1]//2+100]
        peak_pos = (sub_orig==torch.max(sub_orig)).nonzero()
        shift = (peak_pos-torch.tensor([100,100],device=device))[0]
        print(shift)
        matrices[i][0,3] += shift[1]
        matrices[i][1,3] += shift[0]
        model_image, C = project_aunp_model_libtilt(aunps, matrices[i], tilt_series.shape[1:], 8.3, device)
        model_image = model_image * -1.0
        image_product = torch.fft.fft2(tilt_series[i]) * torch.fft.fft2(model_image).conj()
        cc_image = torch.fft.fftshift(torch.fft.ifft2(image_product))
        sub_new = cc_image.real[cc_image.real.shape[0]//2-100:cc_image.real.shape[0]//2+100,cc_image.real.shape[1]//2-100:cc_image.real.shape[1]//2+100]
        if plot:
            axs[0, i].imshow(sub_orig.cpu().detach())
            axs[1, i].imshow(sub_new.cpu().detach())
            # disable axes
            axs[0, i].axis('off')
            axs[1, i].axis('off')
    fig.tight_layout()
    plt.show()
    for tilt in range(len(matrices)):
        for i in range(4):
            arr_str = np.array2string(matrices[tilt][i].cpu().numpy(), precision=12, separator=',', suppress_small=True,max_line_width=1000)
            arr_str = arr_str.replace(' ', '')
            tomogram_infos[tomogram].iloc[tilt,i] = arr_str
    starfile.write(tomogram_infos,str(Path(tomogram_starfile).with_suffix('')) + '_centered.star')


@app.command()
def refine_tilts_using_aunps(
    ctx: FAContext,
    tomogram: str = typer.Argument(...,help="Name of tomogram"),
    aunp_filename: Path = typer.Argument(...,help="Path to aunp file"),
    tomogram_starfile: Optional[Path] = typer.Option(None,help="Path to tomogram star file"),
    plot: bool = typer.Option(False,help="Plot the results")
):
    
    if tomogram_starfile is None:
        tomogram_starfile = ctx.obj.project.settings.FARelionTomoStar
    
    from findingampa.utils.transforms import load_relion_matrices
    from findingampa.utils.projection import load_aunps, project_aunp_model_libtilt
    import mrcfile
    import torch
    import starfile
    import matplotlib.pyplot as plt
    import numpy as np
    from libtilt.transformations import Rx, Ry, Rz, T
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tomogram_infos = starfile.read(tomogram_starfile)

    tomogram_info = tomogram_infos["global"][tomogram_infos["global"]["rlnTomoName"] == tomogram].iloc[0]
    tilt_info = tomogram_infos[tomogram]

    matrices = load_relion_matrices(tilt_info, device)

    aunps = load_aunps(aunp_filename, device)
    tilt_series = torch.as_tensor(mrcfile.read(tomogram_info["rlnTomoTiltSeriesName"]),device=device)


    if plot:
        # Arrange len(matrices) subplots in a
        fig, axs = plt.subplots(2, len(matrices), figsize=(20, 4)) 
    for tilt in range(len(matrices)):
        scores = np.zeros((21,21))

        for a1 in range(21):
            for a2 in range(21):
                a1a = (a1-10) * 0.25
                a2a = (a2-10) * 0.25
                new_matrix = Rx(a1a).to(device).double()[0] @ Ry(a2a).to(device).double()[0] @ matrices[tilt]
                model_image, c = project_aunp_model_libtilt(aunps, new_matrix, tilt_series.shape[1:], 8.3, device)
                model_image = model_image  * -1.0
                image_product = torch.fft.fft2(tilt_series[tilt]) * torch.fft.fft2(model_image).conj()
                cc_image = torch.fft.fftshift(torch.fft.ifft2(image_product))
                sub_new = cc_image.real[cc_image.real.shape[0]//2-100:cc_image.real.shape[0]//2+100,cc_image.real.shape[1]//2-100:cc_image.real.shape[1]//2+100]
                #axs[a1,a2].imshow(sub_new.cpu().detach())
                sub_new -= sub_new.mean()
                sub_new /= sub_new.std()
                score = sub_new.max().item()
                scores[a1,a2] = score
        if plot:
            axs[0,tilt].imshow(scores)
            # Get location of maxima of scores
            m_ind = np.unravel_index(scores.argmax(), scores.shape)
            print(m_ind)
            a1a = (m_ind[0]-10) * 0.25
            a2a = (m_ind[1]-10) * 0.25
            best_matrix = Rx(a1a).to(device).double()[0] @ Ry(a2a).to(device).double()[0] @ matrices[tilt]
            matrices[tilt] = best_matrix
            model_image, c = project_aunp_model_libtilt(aunps, best_matrix, tilt_series.shape[1:], 8.3, device)
            model_image = model_image  * -1.0
            image_product = torch.fft.fft2(tilt_series[tilt]) * torch.fft.fft2(model_image).conj()
            cc_image = torch.fft.fftshift(torch.fft.ifft2(image_product))
            sub_new = cc_image.real[cc_image.real.shape[0]//2-100:cc_image.real.shape[0]//2+100,cc_image.real.shape[1]//2-100:cc_image.real.shape[1]//2+100]
            axs[1,tilt].imshow(sub_new.cpu())
    plt.show()
    for tilt in range(len(matrices)):
        for i in range(4):
            arr_str = np.array2string(matrices[tilt][i].cpu().numpy(), precision=12, separator=',', suppress_small=True,max_line_width=1000)
            arr_str = arr_str.replace(' ', '')
            tomogram_infos[tomogram].iloc[tilt,i] = arr_str
    starfile.write(tomogram_infos,tomogram_starfile.parent / 'tilted.star')
