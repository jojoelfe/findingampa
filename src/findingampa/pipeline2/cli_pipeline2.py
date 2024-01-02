import typer
from rich.prompt import Prompt
app = typer.Typer()

from findingampa.pipeline2.pipeline2_data import Pipeline2Data, Pipeline2TomogramResults, Pipeline2ClusterRegion, Pipeline2TiltInfo
from findingampa.project_management import FAContext, Tomogram
from pathlib import Path
import pandas as pd


@app.command()
def run_pipeline2(
    ctx: FAContext,
    
):
    """Run pipeline 2"""

    from pathlib import Path
    from subprocess import Popen, PIPE, TimeoutExpired, run
    import mrcfile
    from time import sleep
    from datetime import datetime
    from findingampa.pipeline2.pipeline2_steps import read_tilt_info_from_mdoc, run_motioncor2, run_ctffind, run_apply_ctf


    output_path = ctx.obj.project.Settings.FADataDir / "pipeline_data" / "pipeline2"
    output_path.mkdir(parents=True, exist_ok=True)
    data_file = output_path / "pipeline2_data.star"
    if data_file.exists():
        data = Pipeline2Data.read(data_file)
    else:
        data = Pipeline2Data()
        data.write(data_file)

    # Step1: Acquire information
    
    for tomogram in list(ctx.obj.tomograms):
        if tomogram.FAMovies is None:
            print(f"Tomogram {tomogram.FATomogramName} has no movies")
            ctx.obj.tomograms.remove(tomogram)
            continue
        if tomogram.FATomogramName in data.pl2TomogramResults and data.pl2TomogramResults[tomogram.FATomogramName].pl2TiltInfo is not None:
            print(f"Tomogram {tomogram.FATomogramName} already has tilt info")
            continue
        tilt_info = read_tilt_info_from_mdoc(tomogram)
        if tomogram.FATomogramName in data.pl2TomogramResults:
            data.pl2TomogramResults[tomogram.FATomogramName].pl2TiltInfo = tilt_info
        else:
            data.pl2TomogramResults[tomogram.FATomogramName] = Pipeline2TomogramResults(
                pl2TomogramName = tomogram.FATomogramName,
                pl2TiltInfo = tilt_info
            )
        data.write(data_file)
    
    # Step2: Run motioncor2
    for tomogram in ctx.obj.tomograms:
        if data.pl2TomogramResults[tomogram.FATomogramName].pl2MotionCorrDone:
            print(f"Tomogram {tomogram.FATomogramName} already has motioncor2 done")
            continue
        print (f"Running motioncor2 on {tomogram.FATomogramName}")
        success = run_motioncor2(tomogram, tilt_info=data.pl2TomogramResults[tomogram.FATomogramName].pl2TiltInfo,
                                mc2_path=ctx.obj.project.Settings.FAMC2Path, 
                                imod_path=ctx.obj.project.Settings.FAImodPath,
                                pipline_data_path=output_path)
        data.pl2TomogramResults[tomogram.FATomogramName].pl2MotionCorrDone = success
        data.write(data_file)

    # Step2: Run motioncor2
    for tomogram in ctx.obj.tomograms:
        if data.pl2TomogramResults[tomogram.FATomogramName].pl2CtffindDone:
            print(f"Tomogram {tomogram.FATomogramName} already has ctffind done")
            continue
        print (f"Running ctffind on {tomogram.FATomogramName}")
        success = run_ctffind(tomogram, tilt_info=data.pl2TomogramResults[tomogram.FATomogramName].pl2TiltInfo,
                                cistem_path=ctx.obj.project.Settings.FACistemPath,
                                pipline_data_path=output_path)
        data.pl2TomogramResults[tomogram.FATomogramName].pl2CtffindDone = True
        data.write(data_file)
    
    # Step3: Run apply_ctf
    for tomogram in ctx.obj.tomograms:
        run_apply_ctf(tomogram, tilt_info=data.pl2TomogramResults[tomogram.FATomogramName].pl2TiltInfo,
                                cistem_path=ctx.obj.project.Settings.FACistemPath,
                                pipline_data_path=output_path)