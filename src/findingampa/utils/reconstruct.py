from pathlib import Path

def reconstruct_by_relion(
        relion_tomogram_star: Path,
        tomogram_name: str,
        output_path: Path,
        binning: float = 4.0,
        snr: float = 10.0,
        
):
    from subprocess import run
    command = f"relion_tomo_reconstruct_tomogram  --t {relion_tomogram_star} --tn {tomogram_name} --o {output_path} --bin {str(binning)} --j 8 --noctf --SNR {str(snr)}"

    result = run(command,shell=True)
    if result.returncode != 0:
        raise ChildProcessError()
    

def reconstruct_by_libtilt(
        libtilt_tomogram_star: Path,
        tomogram_name: str,
        binning: float = 4.0,
):
    import starfile
    import mrcfile
    import torch
    import einops
    import numpy as np

    from libtilt.rescaling.rescale_fourier import rescale_2d

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tomogram_infos = starfile.read(libtilt_tomogram_star)

    tomogram_info = tomogram_infos["global"][tomogram_infos["global"]["rlnTomoName"] == tomogram_name].iloc[0]
    tilt_info = tomogram_infos[tomogram_name]

    tilt_series = torch.as_tensor(mrcfile.read(tomogram_info["rlnTomoTiltSeriesName"]),device=device)
    tilt_series , _ = rescale_2d(
        image=tilt_series,
        source_spacing=2.5,
        target_spacing=2.5 * binning,
        maintain_center=True
    )
    tilt_series -= einops.reduce(tilt_series, 'tilt h w -> tilt 1 1', reduction='mean')
    tilt_series /= torch.std(tilt_series, dim=(-2, -1), keepdim=True)
    n_tilts, h, w = tilt_series.shape

    

    matrices = tilt_info.iloc[:,[0,1,2,3]].to_numpy()
    matrices = [[s.strip('[]').split(',') for s in matrix] for matrix in matrices]
    matrices = np.array([np.vstack([[float(x) for x in row] for row in matrix]) for matrix in matrices])
    matrices = torch.tensor(matrices,device=device)