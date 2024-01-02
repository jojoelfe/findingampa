from findingampa.project_management import Tomogram
from findingampa.pipeline2.pipeline2_data import Pipeline2Data, Pipeline2TomogramResults, Pipeline2ClusterRegion, Pipeline2TiltInfo
from pathlib import Path



def read_tilt_info_from_mdoc(tomogram: Tomogram):
    from datetime import datetime
    from mdocfile.data_models import Mdoc

    tilt_info_list = []
    for movie_file in tomogram.FAMovies.parent.glob(tomogram.FAMovies.name):
        mdoc_data = Mdoc.from_file(str(movie_file)+ ".mdoc")
        tilt_info = Pipeline2TiltInfo(
            pl2MovieFilename = movie_file,
            pl2Tilt = mdoc_data.section_data[0].TiltAngle,
            pl2AcquisitionTime = datetime.strptime(mdoc_data.section_data[0].DateTime, '%d-%b-%Y  %H:%M:%S'),
            pl2MoviePixelSize = mdoc_data.section_data[0].PixelSpacing,
            pl2FrameNumber= mdoc_data.section_data[0].NumSubFrames,
            pl2AccumulationDose = None,
        )
        tilt_info_list.append(tilt_info)
    #Sort by acquisition time
    tilt_info_list.sort(key=lambda x: x.pl2AcquisitionTime)
    for i, tilt_info in enumerate(tilt_info_list):
        tilt_info.pl2AccumulationDose = i * tomogram.FAExposureDose
    # Sort by tilt
    tilt_info_list.sort(key=lambda x: x.pl2Tilt)
    return tilt_info_list

def run_motioncor2(tomogram: Tomogram, tilt_info: list[Pipeline2TiltInfo], mc2_path: Path, imod_path:Path, pipline_data_path: Path):
    from subprocess import run, DEVNULL

    output_path = pipline_data_path / "motioncor2" / tomogram.FATomogramName
    output_path.mkdir(parents=True, exist_ok=True)

    success = True

    for tilt in tilt_info:
        run(f"""{mc2_path} \\
            -InTiff {tilt.pl2MovieFilename} \\
            -OutMrc {output_path / (tilt.pl2MovieFilename.stem + '.mrc')} \\
            -Gain {tomogram.FAMoviesGain} \\
            -FtBin 2.0 \\
            -FmDose 0.0 \\
            -PixSize {tilt.pl2MoviePixelSize} \\
            -kV 300 \\
            -SplitSum 1 \\
            """,shell=True)
        if not (output_path / (tilt.pl2MovieFilename.stem + '.mrc')).exists():
            success = False
    
    return success

def run_ctffind(tomogram: Tomogram, tilt_info: list[Pipeline2TiltInfo], cistem_path: Path, pipline_data_path: Path):
    from pycistem.config import set_cistem_path
    from pycistem.programs.ctffind import CtffindParameters, run
    import logging
    from rich.logging import RichHandler
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        handlers=[
            RichHandler(),
            #logging.FileHandler(current_output_directory / "log.log")
        ]
    )
    output_path = pipline_data_path / "ctffind" / tomogram.FATomogramName
    output_path.mkdir(parents=True, exist_ok=True)

    set_cistem_path(cistem_path)

    pars = [
        CtffindParameters(
            input_filename=str(tilt.pl2MovieFilename),
            input_is_a_movie=True,
            determine_tilt=True,
            output_diagnostic_filename=str(output_path / (tilt.pl2MovieFilename.stem + '_ctffind.mrc')),
            pixel_size_of_input_image=tilt.pl2MoviePixelSize,
            gain_filename=str(tomogram.FAMoviesGain),
            minimum_resolution=30,
            maximum_resolution=5,
            minimum_defocus=5000,
            maximum_defocus=50000,
            spherical_aberration=0.0
        ) for tilt in tilt_info
    ]
    results = run(pars, num_procs=12, save_output=True, save_output_path="/tmp/ctffind")
    results = sorted(results, key=lambda x: x['parameter_index'])
    print(results)
    for tilt, result  in zip(tilt_info, results):
        tilt.pl2Defocus1 = result['defocus1']
        tilt.pl2Defocus2 = result['defocus2']
        tilt.pl2AstigmatismAngle = result['astigmatism_angle']
        tilt.pl2CtfFitResolution = result['fit_resolution']
        tilt.pl2TiltAxisAngle = result['tilt_axis']
        tilt.pl2TiltAngle = result['tilt_angle']

    return True

def run_apply_ctf(tomogram: Tomogram, tilt_info: list[Pipeline2TiltInfo], cistem_path,  pipline_data_path: Path):
    from pycistem.config import set_cistem_path
    from pycistem.programs.apply_ctf import ApplyCtfParameters, run
    import logging
    from rich.logging import RichHandler
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        handlers=[
            RichHandler(),
            #logging.FileHandler(current_output_directory / "log.log")
        ]
    )
    output_path = pipline_data_path / "applyctf" / tomogram.FATomogramName
    output_path.mkdir(parents=True, exist_ok=True)

    set_cistem_path(cistem_path)

    pars = [
        ApplyCtfParameters(
            input_filename=str(pipline_data_path / "motioncor2" / tomogram.FATomogramName / (tilt.pl2MovieFilename.stem + '_DW.mrc')),
            output_filename=str(output_path / (tilt.pl2MovieFilename.stem + '_DW_deconv.mrc')),
            pixel_size=2*tilt.pl2MoviePixelSize,
            defocus_1=tilt.pl2Defocus1,
            defocus_2=tilt.pl2Defocus2,
            astigmatism_angle=tilt.pl2AstigmatismAngle,
            apply_wiener_filter=True,
            wiener_filter_falloff_frequency=100,
            wiener_filter_falloff_fudge_factor=1.0,
            wiener_filter_scale_fudge_factor=1.0,
            wiener_filter_high_pass_radius=200.0
        ) for tilt in tilt_info
    ]
    print(pars)
    run(pars, num_procs=12, save_output=True, save_output_path="/tmp/ctffind")
    

    return True


def create_stack():    
    # Create stack
    stack_filename = pipline_data_path / "stacks" / tomogram.FATomogramName / (tomogram.FATomogramName + ".mrc")
    stack_filename.parent.mkdir(parents=True, exist_ok=True)
    run(f"""{imod_path}/newstack \\
        {' '.join([str(output_path / (tilt.pl2MovieFilename.stem + '_DW.mrc')) for tilt in tilt_info])} \\
        {stack_filename} \\
        """, shell=True)
    
    return [stack_filename]