import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 
import typer
import starfile
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
#from findingampa.pipeline1.cli_pipeline1 import app as pipeline1_app
#from findingampa.pipeline2.cli_pipeline2 import app as pipeline2_app
#from findingampa.pipeline3.cli_pipeline3 import app as pipeline3_app
from findingampa.project_management import Project, FAContext, FAGlobal, Tomogram, RelionTomogram
from typing import Optional
from rich import print

app = typer.Typer(
)

#app.registered_commands += pipeline1_app.registered_commands
#app.registered_commands += pipeline2_app.registered_commands
#app.registered_commands += pipeline3_app.registered_commands

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
def update_project(
    ctx: FAContext,
):
    ctx.obj.project.data = pd.DataFrame([x.model_dump() for x in ctx.obj.tomograms])
    starfile.write(ctx.obj.project.model_dump(),Path.cwd() / "finding_ampa_project.star")

@app.command()
def new_project(
    tomogram_star: Path = typer.Argument(...,help="Path to tomogram star file")
):
    """Create a new project file"""
    if not Path.cwd().joinpath("default_pipeline.star").exists():
        typer.echo("This does not appear to be a relion project folder, please run finding_ampa there.")
        raise typer.Exit()
    finding_ampa_folder = Path.cwd() / "finding_ampa"
    finding_ampa_folder.mkdir(exist_ok=True)
    project_file = finding_ampa_folder / "finding_ampa_project.star"
    if project_file.exists():
        typer.echo("Project file already exists")
        raise typer.Exit()
    project = Project()
    project.settings.FARelionTomoStar = tomogram_star
    tomograms = starfile.read(tomogram_star)
    for i, tomogram in tomograms["global"].iterrows():
        project.data.append(Tomogram(FATomogramName=tomogram["rlnTomoName"]))
    project.write(project_file)

@app.command()
def create_tomograms():
    from findingampa.utils.reconstruct import run_tilt
    from findingampa.utils.imod import get_exclude_list, get_align_angle_offset, get_tomopitch_angle_offset, get_full_image
    try:
        tomo_name = list(Path.cwd().glob("*.rawtlt"))[0].stem
    except IndexError:
        typer.echo("No rawtlt file found")
        raise typer.Exit()
    print(f" Creating tomograms for {tomo_name}")
    exclude_list = get_exclude_list()
    print(f"Excluding tilts {exclude_list}")
    if get_align_angle_offset() > 0.0:
        print("Align angle offset is not 0.0, please reset to 0.0 and recreate aligned stack and doseweight")
        raise typer.Exit()
    angle_offset = get_tomopitch_angle_offset()
    full_image = get_full_image()
    print(f"Creating WBP untilted tomogram {tomo_name}_full_rec_BP.mrc")
    run_tilt(
        f"{tomo_name}_ali.mrc",
        f"{tomo_name}_full_rec_BP.mrc",
        f"{tomo_name}.tlt",
        exclude_list=exclude_list,
        thickness=3000,
        full_image=full_image
    )
    print(f"Creating WBP tilted tomogram {tomo_name}_full_rec_BP_flat.mrc")
    run_tilt(
        f"{tomo_name}_ali.mrc",
        f"{tomo_name}_full_rec_BP_flat.mrc",
        f"{tomo_name}.tlt",
        exclude_list=exclude_list,
        angle_offset=angle_offset,
        thickness=1500,
        full_image=full_image
    )
    print(f"Creating SIRT untilted tomogram {tomo_name}_full_rec_SIRT.mrc")
    run_tilt(
        f"{tomo_name}_ali.mrc",
        f"{tomo_name}_full_rec_SIRT.mrc",
        f"{tomo_name}.tlt",
        SIRT_iterations=11,
        exclude_list=exclude_list,
        thickness=3000,
        full_image=full_image
    )
    print(f"Creating SIRT tilted tomogram {tomo_name}_full_rec_SIRT_flat.mrc")
    run_tilt(
        f"{tomo_name}_ali.mrc",
        f"{tomo_name}_full_rec_SIRT_flat.mrc",
        f"{tomo_name}.tlt",
        SIRT_iterations=11,
        exclude_list=exclude_list,
        angle_offset=angle_offset,
        thickness=1500,
        full_image=full_image
    )

@app.command()
def annotate_membranes():
    try:
        tomo_name = list(Path.cwd().glob("*.rawtlt"))[0].stem
    except IndexError:
        typer.echo("No rawtlt file found")
        raise typer.Exit()
    
    MEMBRAIN_MODEL_PATH = None
    potential_membrain_model_paths = [
        Path("/goliath/processing/Gouaux/CJS/software/membrain-seg/models/MemBrain_seg_v10_alpha.ckpt"),
        Path("/home/elferich/ProcessingJE/manuscript01/relion_reconstruction/finding_ampa/static/MemBrain_seg_v10_alpha.ckpt")
    ]
    for path in potential_membrain_model_paths:
        if path.exists():
            MEMBRAIN_MODEL_PATH = path
            break

    from membrain_seg.segmentation.segment import segment

    membrain_output_path = Path("membrain" )
    membrain_output_path.mkdir(parents=True, exist_ok=True)
    
    segmentation_file = segment(
        tomogram_path=f"{tomo_name}_full_rec_BP_flat.mrc",
        ckpt_path=MEMBRAIN_MODEL_PATH,
        out_folder=membrain_output_path,
        segmentation_threshold=0.0,
        store_probabilities=True
    )
    print(f"Segmentation saved to {segmentation_file}")

@app.command()
def annotate_aunps():
    try:
        tomo_name = list(Path.cwd().glob("*.rawtlt"))[0].stem
    except IndexError:
        typer.echo("No rawtlt file found")
        raise typer.Exit()
    potential_blender_paths= [
        Path("/home/elferich/Downloads/blender-4.2.0-alpha+main.bace4c9a293c-linux.x86_64-release/blender"),
        Path("/groups/elferich/Downloads/blender-4.2.0-alpha+main.6c78f7974098-linux.x86_64-release/blender"),
        Path("/goliath/sw/blender/blender-4.2.0-alpha+main.2f67c45b650b-linux.x86_64-release/blender")
    ]
    BLENDER_PATH = None
    for path in potential_blender_paths:
        if path.exists():
            BLENDER_PATH = path
            break
    if BLENDER_PATH is None:
        typer.echo("Blender not found")
        raise typer.Exit()
    template_file = Path(__file__).parent / "aunp_template.blend"
    aunp_folder = Path("aunps")
    aunp_folder.mkdir(parents=True, exist_ok=True)
    #Copy template file into aunp_folder
    import shutil
    shutil.copy(template_file, aunp_folder)

    import os
    os.system(f"{BLENDER_PATH} {aunp_folder / template_file.name}  -P {Path(__file__).parent / 'bpy_annotate_aunps.py'} -- --tomogram {tomo_name}_full_rec_BP_flat.mrc")   

    # Create model files
    from subprocess import Popen, PIPE
    from findingampa.utils.imod import get_tomopitch_angle_offset
    coordinates = Path("aunps").glob("*.txt")
    tilt_angle = get_tomopitch_angle_offset()
    for coordinate_file in coordinates:
        pro1 = Popen(f"point2model -sc -sp 2 -im {tomo_name}_full_rec_BP_flat.mrc {coordinate_file} {coordinate_file.with_suffix('.mod')}", stdout=PIPE, stderr=PIPE, shell=True)
        pro1.communicate()
        print(f"imodtrans  -ry {tilt_angle} -tz 187.5 {coordinate_file.with_suffix('.mod')} {str(coordinate_file)[:-4] + '_orig.mod'}")
        pro2 = Popen(f"imodtrans  -ry {tilt_angle} -tz 187.5 {coordinate_file.with_suffix('.mod')} {str(coordinate_file)[:-4] + '_orig.mod'}", stdout=PIPE, stderr=PIPE, shell=True)
        pro2.communicate()