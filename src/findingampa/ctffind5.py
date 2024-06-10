import sys
import os
import subprocess
from pathlib import Path
import typer
from rich import print
from pydantic import BaseModel
from typing import Optional, List
from enum import Enum
from mdocfile.data_models import Mdoc
from rich.progress import track

class CTfFind5Input(BaseModel):
    input_filename: str
    input_is_a_movie: str = "No"
    output_diagnostic_filename: str = "diag.mrc"
    pixel_size_of_input_image: float = 2.5
    acceleration_voltage: float = 300.0
    spherical_aberration: float = 0.01
    amplitude_contrast: float = 0.07
    size_of_amplitude_spectrum_to_compute: int = 512
    minimum_resolution: float = 50.0
    maximum_resolution: float = 8.0
    minimum_defocus: float = 5000.0
    maximum_defocus: float = 50000.0
    defocus_search_step: float = 100.0
    know_astigmatism: str = "No"
    slower_search: str = "No"
    restraint_on_astigmatism: str = "No"
    find_additional_phase_shift: str = "No"
    determine_sample_tilt: str = "Yes"
    determine_sample_thickness: str = "No"
    expert_options: Optional[str] = "No"

app = typer.Typer()


@app.command()
def run_multiple(
    max_workers: int = typer.Option(20, help="Number of workers to use"),
    sa: float = typer.Option(0.01, help="Spherical aberration in mm"),
    folders: list[Path] = typer.Argument(None)
    ):
    from concurrent.futures import ThreadPoolExecutor
    print(folders)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Start the load operations and mark each future with the URL
        future_to_tilt_series = {executor.submit(run_single_tomogram, tilt_series_folder=folder, sa=sa, run_ctffind5=True): folder for folder in folders}
        for future in track(future_to_tilt_series, description="Processing tilt series"):
            row = future_to_tilt_series[future]
            try:
                future.result()
            except Exception as exc:
                print(f"Processing {row['Tomogram name']} generated an exception: {exc}")




@app.command()
def run_csv(
    csv_file: Path = typer.Argument(..., help="Path to the csv file"),
    run_ctffind5: bool = typer.Option(True, help="Run ctffind5"),
    max_workers: int = typer.Option(48, help="Number of workers to use"),
):
    import pandas as pd
    from rich import print

    from concurrent.futures import ThreadPoolExecutor
    df = pd.read_csv(csv_file)
    df = df[df["Tomogram name"].notna()]
    print(df)
    
    existing_tilt_series = df[df["Tomogram name"].apply(lambda x: (Path("/scratch/pompeii/elferich/gouaux_tomo/ProcessedData/tomograms/") / str(x) / f"{x}.mrc").exists())]
    print(existing_tilt_series["Tomogram name"])
    # Setup the ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Start the load operations and mark each future with the URL
        future_to_tilt_series = {executor.submit(run_single_tomogram, tilt_series_folder=Path("/scratch/pompeii/elferich/gouaux_tomo/ProcessedData/tomograms/") / row["Tomogram name"], sa=row['Cs'], run_ctffind5=run_ctffind5): row for index, row in existing_tilt_series.iterrows()}
        for future in track(future_to_tilt_series, description="Processing tilt series"):
            row = future_to_tilt_series[future]
            try:
                future.result()
            except Exception as exc:
                print(f"Processing {row['Tomogram name']} generated an exception: {exc}")

@app.command()
def analyze_csv(
    csv_file: Path = typer.Argument(..., help="Path to the csv file"),
):
    import pandas as pd
    from rich import print

    from concurrent.futures import ThreadPoolExecutor
    df = pd.read_csv(csv_file)
    
    df = df[df["Tomogram name"].notna()]
    print(df)
    
    existing_tilt_series = df[df["Tomogram name"].apply(lambda x: (Path("/scratch/pompeii/elferich/gouaux_tomo/ProcessedData/tomograms/") / str(x) / f"{x}.mrc").exists())]
  
    
    # Setup the ThreadPoolExecutor
    for index, row in existing_tilt_series.iterrows():
        try:
            analyze_single_tomogram(tilt_series_folder=Path("/scratch/pompeii/elferich/gouaux_tomo/ProcessedData/tomograms/") / row["Tomogram name"])
            print(f"Processed {row['Tomogram name']}")
        except Exception as exc:
            print(f"Processing {row['Tomogram name']} generated an exception: {exc}")
            Path(f"/scratch/pompeii/elferich/gouaux_tomo/ProcessingJE/ctffind5/{row['Tomogram name']}").mkdir(parents=True, exist_ok=True)
            with open(f"/scratch/pompeii/elferich/gouaux_tomo/ProcessingJE/ctffind5/{row['Tomogram name']}/ctffind5_fit.csv", "w") as f:
                f.write(f"{row['Tomogram name']},0,0,0,0,0\n")
    
@app.command()
def analyze_multiple(
    max_workers: int = typer.Option(20, help="Number of workers to use"),
    sa: float = typer.Option(0.01, help="Spherical aberration in mm"),
    folders: list[Path] = typer.Argument(None)
    ):
    for folder in folders:
        try:
            analyze_single_tomogram(folder)
            print(f"Processed {folder.name}")
        except Exception as exc:
            print(f"Processing {folder.name} generated an exception: {exc}")

@app.command()
def run_single_tomogram(
    tilt_series_folder: Path = typer.Argument(..., help="Path to the folder with teh tiltseries"),
    sa: float = typer.Option(0.01, help="Spherical aberration in mm"),
    run_ctffind5: bool = typer.Option(True, help="Run ctffind5"),
):
    
    from subprocess import Popen, PIPE
    

    output_path = Path("/goliath/processing/Gouaux/CJS/BestTomo/ProcessingJE/ctffind5")
    cistem_path = Path("/goliath/processing/Gouaux/CJS/BestTomo/ProcessingJE")

    output_path = output_path / tilt_series_folder.name
    output_path.mkdir(parents=True, exist_ok=True)
    pars = CTfFind5Input(
         input_filename=str(tilt_series_folder / f"{tilt_series_folder.name}.mrc"),
         output_diagnostic_filename=str(output_path / 'diagnostic_output.mrc'),
         spherical_aberration=sa
    )
    if not Path(pars.output_diagnostic_filename).exists() and run_ctffind5:      
        proc = Popen(cistem_path / "ctffind",  stdin=PIPE)
        proc.communicate(input=str.encode('\n'.join(map(str,list(pars.model_dump().values())))))

@app.command()
def analyze_single_tomogram(
    tilt_series_folder: Path = typer.Argument(..., help="Path to the folder with teh tiltseries"),
):
    """
    # rotation matrix transform, find the 0 tilt initial axis direction and tilt angle
    # by Lingli Kong @ Grigorieff lab

    # minor changes by Johannes Elferich @ Grigorieff lab to make style consistent
    """
    import numpy as np
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation as R


    output_path = Path("/goliath/processing/Gouaux/CJS/BestTomo/ProcessingJE/ctffind5")

    output_path = output_path / tilt_series_folder.name
    ctffind5_info_overall = np.loadtxt(output_path / 'diagnostic_output.txt')
    ctffind5_info = ctffind5_info_overall[:,[0,7,8]]
   

    image_no=len(ctffind5_info[:,0])

    #microscope angle system:
    rawtlt= np.loadtxt(tilt_series_folder / f"{tilt_series_folder.name}.rawtlt")
    #rawtlt = rawtlt.to_numpy()
    #rawtlt = rawtlt[:,0]
    tem_info=np.zeros(np.shape(ctffind5_info))
    tem_info[:,0]=ctffind5_info[:,0] #index
    tem_info[:,1]=0 #unknown
    tem_info[:,2]=rawtlt


    ctf_rotations = [R.from_euler('zxz',[-ctffind5_info[i,1],ctffind5_info[i,2],ctffind5_info[i,1]],degrees=True) for i in range(image_no)]

    def optim_function(x,index=None):
        global outliers
        phi_zero, theta_zero, phi_tem = x
        res = []
        if index is None:
            index = range(image_no)
        for i in index:
            zero_rot = R.from_euler('zxz',[-phi_zero,theta_zero,phi_zero],degrees=True)
            tem_rot = R.from_euler('zxz',[-phi_tem,rawtlt[i],phi_tem],degrees=True)
            ctf_vec = ctf_rotations[i].apply([0,0,1])
            fit_vec = (tem_rot * zero_rot).apply([0,0,1])
            # Calculate angle between vectors ctf_vec and fit_vec
            res.append(np.linalg.norm(ctf_vec-fit_vec))
        rmse = np.sqrt(np.mean(np.array(res)**2))
        outliers = np.where(np.array(res)/rmse > 3.0)[0]
        if np.max(np.array(res)/rmse) > 3.0:
            
            return optim_function(x, np.where(np.array(res)/rmse < 3.0)[0])    
        return rmse

    def optim_function_get_outliers(x,index=None):
        global outliers
        phi_zero, theta_zero, phi_tem = x
        res = []
        if index is None:
            index = range(image_no)
        for i in index:
            zero_rot = R.from_euler('zxz',[-phi_zero,theta_zero,phi_zero],degrees=True)
            tem_rot = R.from_euler('zxz',[-phi_tem,rawtlt[i],phi_tem],degrees=True)
            ctf_vec = ctf_rotations[i].apply([0,0,1])
            fit_vec = (tem_rot * zero_rot).apply([0,0,1])
            # Calculate angle between vectors ctf_vec and fit_vec
            res.append(np.linalg.norm(ctf_vec-fit_vec))
        rmse = np.sqrt(np.mean(np.array(res)**2))
        outliers = np.where(np.array(res)/rmse > 2.0)[0]
        return rmse, outliers


    from scipy.optimize import minimize

    sol = minimize(optim_function, [0,0,0])


    phi_zero, theta_zero, phi_tem = sol.x
    rmse, outliers = optim_function_get_outliers(sol.x)
    # JE Plotting code

    exp_rot = [ R.from_euler('zxz',[-phi_tem,rawtlt[i],phi_tem],degrees=True) * R.from_euler('zxz',[-phi_zero,theta_zero,phi_zero],degrees=True) for i in range(image_no)]

    exp_tilt = [r.as_euler('zxz',degrees=True)[1] for r in exp_rot]
    exp_axis = [r.as_euler('zxz',degrees=True)[2] for r in exp_rot]
    exp_axis = [a+360 if a < 0 else a for a in exp_axis]


    # genrate two plots, One for fitted alues one for errors. Te error plot should be below and have 1/4 of the height. The figure should be 8.85 cm wide.
        
    fig,(ax_tilt, ax_axis_angle, ax_error) = plt.subplots(3,1,figsize=(3.5,3.5),sharex=True,gridspec_kw={'height_ratios':[2,2,1]})  
    ctffind5_data = ctffind5_info
    outlier_mask = np.in1d(range(ctffind5_data.shape[0]), outliers)
    no_outlier_mask = np.invert(outlier_mask)

    ax_tilt.plot(rawtlt,exp_tilt,linestyle='--',color='r')
    ax_tilt.scatter(rawtlt[no_outlier_mask],ctffind5_data[no_outlier_mask,2],marker='o')
    if len(outliers) > 0:
        ax_tilt.scatter(ctffind5_data[outliers,0],ctffind5_data[outliers,2],marker='o',color='grey')#,label='outliers')

    ax_axis_angle.plot(rawtlt,exp_axis,linestyle='--',color='r',label='Overall tilt model')
    ax_axis_angle.scatter(rawtlt[no_outlier_mask],ctffind5_info[no_outlier_mask,1],marker='x',label='Single tilt measurement')
    if len(outliers) > 0:
        ax_axis_angle.scatter(rawtlt[outliers],ctffind5_data[outliers,1],color='grey',marker='x')#,label='outliers')


    ax_error.scatter(rawtlt[no_outlier_mask],ctffind5_info[no_outlier_mask,2]-np.array(exp_tilt)[no_outlier_mask],marker='o')
    ax_error.scatter(rawtlt[no_outlier_mask],ctffind5_info[no_outlier_mask,1]-np.array(exp_axis)[no_outlier_mask],marker='x',color='k')
    ax_error.hlines(0,tem_info[0,2],tem_info[-1,2],'k',linestyles='-')


    fig.subplots_adjust(hspace=0.05)
    ax_tilt.set_ylabel('Tilt angle (°)')
    ax_axis_angle.set_ylabel('Tilt axis angle (°)')
    ax_error.set_ylabel('Error (°)')
    ax_error.set_xlabel('Nominal stage tilt angle (°)')
    ax_error.set_ylim(-12,12)
    plt.tight_layout()
    plt.savefig(output_path / 'ctffind5_fit.png')
    with open(output_path / 'ctffind5_fit.csv', 'w') as f:
        #f.write(f"{phi_zero},")
        #f.write(f"theta_zero: {theta_zero}\n")
        f.write(f"{tilt_series_folder.name},{phi_tem},{rmse},{len([a+1 for a in outliers])},{np.mean(np.array(sorted(ctffind5_info_overall[:,6])[:5]))},{np.sum(ctffind5_info_overall[:,6] > 12.5)}\n")
        #f.write(f"outliers: {len([a+1 for a in outliers])}\n")
    
        #f.write(f"rmse: {rmse}\n")
        #f.write(f"average res {np.mean(ctffind5_info_overall[:,6])}\n")
        #f.write(f"high res {np.min(ctffind5_info_overall[:,6])}\n")
        #f.write(f"res_outliers {[a+1 for a in np.where(ctffind5_info_overall[:,6] > 12.5)]}\n")
    plt.close()

    
if __name__ == "__main__":
    app()
