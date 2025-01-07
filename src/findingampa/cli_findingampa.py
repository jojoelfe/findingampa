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
    from findingampa.utils.cryocare import apply_cryocare
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
    #print(f"Creating WBP untilted tomogram {tomo_name}_full_rec_BP.mrc")
    #run_tilt(
    #    f"{tomo_name}_ali.mrc",
    #    f"{tomo_name}_full_rec_BP.mrc",
    #    f"{tomo_name}.tlt",
    #    exclude_list=exclude_list,
    #    thickness=3000,
    #    full_image=full_image
    #)
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
    create_bin_3dctf_tomogram(bin_factor=4, 
                              FakeSirtIterations=0, 
                              num_gpus=1, 
                              never_erase=False, 
                              delete_aunps=False, 
                              custom_xf=None, 
                              flatten=True, 
                              tilt_parent=False)
    tilt_parent = False
    if Path(f"../{tomo_name}_EVN.mrc").exists():
        tilt_parent = True
    elif Path(f"{tomo_name}_EVN.mrc").exists():
        tilt_parent = False
    else:
        print(f"No even/odd stacks found")
        return
    print(f"Creating EVN tilted tomogram")
    create_bin_3dctf_tomogram(bin_factor=4, 
                              FakeSirtIterations=0, 
                              num_gpus=1, 
                              never_erase=False, 
                              delete_aunps=False, 
                              custom_xf=None, 
                              custom_suffix="_EVN", 
                              flatten=True, 
                              tilt_parent=tilt_parent)
    print(f"Creating ODD tilted tomogram")
    create_bin_3dctf_tomogram(bin_factor=4, 
                              FakeSirtIterations=0, 
                              num_gpus=1, 
                              never_erase=False, 
                              delete_aunps=False, 
                              custom_xf=None, 
                              custom_suffix="_ODD", 
                              flatten=True, 
                              tilt_parent=tilt_parent)
    denoised = apply_cryocare(tomo_name)
    
@app.command()
def ddw():
    import mrcfile
    try:
        tomo_name = list(Path.cwd().glob("*.rawtlt"))[0].stem
    except IndexError:
        typer.echo("No rawtlt file found")
        raise typer.Exit()
    from subprocess import Popen, PIPE

    with mrcfile.open(f"{tomo_name}_full_rec_BP_3DCTF_BIN4_EVN.mrc", mode="r+") as f:
        dimensions = f.header["nx"], f.header["ny"], f.header["nz"]
    
    ratio = dimensions[0]/dimensions[1]
    if abs(ratio - 1) > 0.1:
        print("Tomogram is not square, assuming K3")
        check_point_file = "ddw_k3.ckpt"
    else:
        print("Tomogram is square, assuming Falcon")
        check_point_file = "20241101.ckpt"
    
    Popen(f"scp *BIN4_EVN.mrc *BIN4_ODD.mrc exahead1:/home/exacloud/gscratch/gouaux/CJS/deepdewedge/data/", shell=True).wait()
    Popen(f"ssh exahead1 srun -p gouaux --mem 32G --time 5:00:00 -c 10 --gres gpu:a40:1 /home/exacloud/gscratch/gouaux/CJS/deepdewedge/ddw_env/bin/ddw refine-tomogram --tomo0-files /home/exacloud/gscratch/gouaux/CJS/deepdewedge/data/{tomo_name}_full_rec_BP_3DCTF_BIN4_EVN.mrc  --tomo1-files /home/exacloud/gscratch/gouaux/CJS/deepdewedge/data/{tomo_name}_full_rec_BP_3DCTF_BIN4_ODD.mrc --subtomo-size 96 --mw-angle 84 --subtomo-overlap 32 --batch-size 5 --model-checkpoint-file /home/exacloud/gscratch/gouaux/CJS/deepdewedge/checkpoints/{check_point_file} --output-dir /home/exacloud/gscratch/gouaux/CJS/deepdewedge/data/ --num-workers 10 --gpu 0", shell=True).wait()
    Popen(f"scp exahead1:/home/exacloud/gscratch/gouaux/CJS/deepdewedge/data/{tomo_name}*_refined.mrc {tomo_name}_full_rec_BP_3DCTF_BIN4_ddw.mrc", shell=True).wait()


@app.command()
def new_annotate_aunps(
    reset: bool = False
):
    from findingampa.utils.assets import FA_ASSETS
    try:
        tomo_name = list(Path.cwd().glob("*.rawtlt"))[0].stem
    except IndexError:
        typer.echo("No rawtlt file found")
        raise typer.Exit()
    
    BLENDER_PATH = FA_ASSETS / "blender" / "4_3" / "blender"
    
    if not BLENDER_PATH.exists():
        typer.echo("Blender not found")
        raise typer.Exit()
    template_file = Path(__file__).parent / "new_aunp_template.blend"
    
    aunp_folder = Path("aunps")
    aunp_folder.mkdir(parents=True, exist_ok=True)
    annotated_membranes_folder = Path("annotated_membranes")
    annotated_membranes_folder.mkdir(parents=True, exist_ok=True)
    instance_template_file = aunp_folder / "new_aunp_template.blend"
    #Copy template file into aunp_folder
    import shutil
    if instance_template_file.exists() and not reset:
        should_setup = False
    else:
        shutil.copy(template_file, aunp_folder)
        should_setup = True

    import os
    print(should_setup)
    os.system(f"{BLENDER_PATH} {aunp_folder / template_file.name}  -P {Path(__file__).parent / 'bpy_new_aunp_positions.py'} -- --tomogram {tomo_name}_full_rec_BP_flat.mrc {'--setup' if should_setup else ''} --findingampa {Path(__file__).parent.parent}")   



@app.command()
def annotate_membranes():
    from findingampa.utils.assets import FA_ASSETS
    import mrcfile
    try:
        tomo_name = list(Path.cwd().glob("*.rawtlt"))[0].stem
    except IndexError:
        typer.echo("No rawtlt file found")
        raise typer.Exit()
    
    MEMBRAIN_MODEL_PATH = FA_ASSETS / "membrain" / "MemBrain_seg_v10_alpha.ckpt"

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
    smooth_membranes(segmentation_file)
    ddw = Path(f"{tomo_name}_full_rec_BP_3DCTF_BIN4_ddw.mrc")
    if ddw.exists():
        with mrcfile.open(ddw, mode="r+") as f:
                f.voxel_size = 10.0
        segmentation_file = segment(
            tomogram_path=str(ddw),
            ckpt_path=MEMBRAIN_MODEL_PATH,
            out_folder=membrain_output_path,
            segmentation_threshold=0.0,
            store_probabilities=True
        )
        print(f"Segmentation saved to {segmentation_file}")
        smooth_membranes(segmentation_file)
        return
    denoised = Path(f"{tomo_name}_full_rec_BP_3DCTF_BIN4_denoised.mrc")
    if denoised.exists():
        segmentation_file = segment(
            tomogram_path=str(denoised),
            ckpt_path=MEMBRAIN_MODEL_PATH,
            out_folder=membrain_output_path,
            segmentation_threshold=0.0,
            store_probabilities=True
        )
        print(f"Segmentation saved to {segmentation_file}")
        smooth_membranes(segmentation_file)

@app.command()
def smooth_membranes(segmentation_file = None):
    if segmentation_file is None:
        try:
            tomo_name = list(Path.cwd().glob("*.rawtlt"))[0].stem
        except IndexError:
            typer.echo("No rawtlt file found")
            raise typer.Exit()
        segmentation_file = Path(f"membrain/{tomo_name}_full_rec_BP_flat_MemBrain_seg_v10_alpha.ckpt_segmented.mrc")
    from subprocess import Popen, PIPE
    pro1 = Popen(f"trimvol -c 0,1 {segmentation_file} membrain/scaled.mrc", stdout=PIPE, stderr=PIPE, shell=True)
    pro1.communicate()
    pro2 = Popen(f"clip smooth -l 4.0 -m float membrain/scaled.mrc {segmentation_file[:-4]}_smooth.mrc", stdout=PIPE, stderr=PIPE, shell=True)
    pro2.communicate()

@app.command()
def annotate_ampa_positions():
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
    template_file = Path(__file__).parent / "ampa_position_template.blend"
    
    ampa_folder = Path("ampa_positions")
    ampa_folder.mkdir(parents=True, exist_ok=True)
    annotated_membranes_folder = Path("annotated_membranes")
    annotated_membranes_folder.mkdir(parents=True, exist_ok=True)
    instance_template_file = ampa_folder / "ampa_position_template.blend"
    #Copy template file into aunp_folder
    import shutil
    if instance_template_file.exists():
        should_setup = False
    else:
        shutil.copy(template_file, ampa_folder)
        should_setup = True

    import os
    print(should_setup)
    os.system(f"{BLENDER_PATH} {ampa_folder / template_file.name}  -P {Path(__file__).parent / 'bpy_ampa_positions.py'} -- --tomogram {tomo_name}_full_rec_BP_flat.mrc {'--setup' if should_setup else ''} --findingampa {Path(__file__).parent.parent}")   

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


@app.command()
def add_to_relion_star(
    relion_star_path: Path,
):
    if not relion_star_path.exists():
        relion_star = pd.DataFrame(columns=["rlnTomoName", "rlnTomoTiltSeriesName", "rlnTomoImportCtfPlotterFile", "rlnTomoImportImodDir", "rlnTomoImportFractionalDose", "rlnTomoImportOrderList", "rlnSphericalAberration", "rlnTomoImportCulledFile"])
    else:
        relion_star = starfile.read(relion_star_path)
    try:
        tomo_name = list(Path.cwd().glob("*.rawtlt"))[0].stem
    except IndexError:
        typer.echo("No rawtlt file found")
        raise typer.Exit()
    rlnTomoName = tomo_name
    rlnTomoTiltSeriesName = Path(tomo_name+'.mrc').absolute()
    rlnTomoImportCtfPlotterFile = Path(tomo_name+'.defocus').absolute()
    rlnTomoImportImodDir = Path.cwd().absolute()
    rlnTomoImportFractionalDose = 4.55
    rlnTomoImportOrderList = Path.cwd().parent.parent.parent.parent / "ProcessedData" / "tomograms" / tomo_name / (tomo_name + ".csv")
    rlnSphericalAberration = 2.7
    rlnTomoImportCulledFile = Path(tomo_name+'_relionculled.mrc').absolute()
    relion_star.loc[len(relion_star)] = [rlnTomoName, rlnTomoTiltSeriesName, rlnTomoImportCtfPlotterFile, rlnTomoImportImodDir, rlnTomoImportFractionalDose, rlnTomoImportOrderList, rlnSphericalAberration, rlnTomoImportCulledFile]
    starfile.write(relion_star,relion_star_path)
    print(f"Added {tomo_name} to {relion_star_path}")
    #Change THICKNESS in tilt.com
    tilt_com = Path.cwd() / "tilt.com"
    with open(tilt_com, "r") as f:
        lines = f.readlines()
    # Backup tilt.com
    with open(tilt_com.with_suffix(".bakfa"), "w") as f:
        f.writelines(lines)
    for i, line in enumerate(lines):
        if "THICKNESS" in line:
            lines[i] = f"THICKNESS 3000\n"
    with open(tilt_com, "w") as f:
        f.writelines(lines)
    print(f"Changed THICKNESS in {tilt_com}")


@app.command()
def add_ampa_positions_to_relion_star(
    positions_set: str,
    relion_star_path: Path
):
    if not relion_star_path.exists():
        relion_star = pd.DataFrame(columns=["rlnTomoName", "rlnTomoImportParticleFile"])
    else:
        relion_star = starfile.read(relion_star_path)
    try:
        tomo_name = list(Path.cwd().glob("*.rawtlt"))[0].stem
    except IndexError:
        typer.echo("No rawtlt file found")
        raise typer.Exit()
    rlnTomoName = tomo_name
    rlnTomoImportParticleFile = Path.cwd() / "ampa_positions" / f"{positions_set}.star"
    if not rlnTomoImportParticleFile.exists():
        typer.echo(f"File {rlnTomoImportParticleFile} does not exist")
        raise typer.Exit()
    relion_star.loc[len(relion_star)] = [rlnTomoName, rlnTomoImportParticleFile]
    starfile.write(relion_star,relion_star_path)

@app.command()
def create_bin_3dctf_tomogram(
    bin_factor: int = typer.Option(2),
    FakeSirtIterations: int = typer.Option(8),
    num_gpus: int = typer.Option(1),
    never_erase: bool = False,
    delete_aunps: bool = False,
    custom_xf: Optional[Path] = None,
    custom_suffix: str = "",
    flatten: bool = False,
    tilt_parent: bool = False
):
    from findingampa.utils.imod import check_ccderaser, get_tomopitch_angle_offset
    from subprocess import Popen, PIPE
    import mrcfile
    import shutil
    try:
        tomo_name = list(Path.cwd().glob("*.rawtlt"))[0].stem
    except IndexError:
        typer.echo("No rawtlt file found")
        raise typer.Exit()
    
    remove_beads = check_ccderaser(tomo_name)
    if never_erase:
        remove_beads = False
    print(f"Remove beads: {remove_beads}")
    # Make dose file local in mtffilter.com
    mtffilter_com = Path.cwd() / "mtffilter.com"
    with open(mtffilter_com, "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "DoseWeightingFile" in line:
            lines[i] = f"DoseWeightingFile {tomo_name}.dose\n"
    with open(mtffilter_com, "w") as f:
        f.writelines(lines)
    #Change THICKNESS in tilt.com
    tilt_com = Path.cwd() / "tilt.com"
    with open(tilt_com, "r") as f:
        lines = f.readlines()
    # Backup tilt.com
    with open(tilt_com.with_suffix(".bakfa"), "w") as f:
        f.writelines(lines)
    for i, line in enumerate(lines):
        if flatten and "OFFSET" in line:
            lines[i] = f"OFFSET {get_tomopitch_angle_offset()}\n"
            #print(get_align_angle_offset())
        if "THICKNESS" in line:
            lines[i] = f"THICKNESS 1500\n" if flatten else f"THICKNESS 3000\n"
        
    newlines = []
    for i, line in enumerate(lines):
        if "fakesirtiterations" not in line.lower():
            newlines.append(line)
    lines = newlines
    if FakeSirtIterations > 0:
        lines.insert(20,f"FakeSirtIterations {FakeSirtIterations}\n")
    with open(tilt_com, "w") as f:
        f.writelines(lines)
    if delete_aunps:
        models = list(Path("aunps/").glob("SavedPicks_*_orig.mod"))
        if len(models) > 1:
            pro = Popen(f"imodjoin -n {' '.join([str(m) for m in models])} aunps/combined_savedpicks.mod", shell=True)
            pro.wait()
        else:
            shutil.copy(models[0],"aunps/combined_savedpicks.mod")
        pro = Popen(f"submfg tilt.com",shell=True)
        pro.wait()
        pro = Popen(f"imodtrans -i {tomo_name}_full_rec_BP.mrc aunps/combined_savedpicks.mod aunps/combined_savedpicks_asim.mod",shell=True)
        pro.wait()
        pro = Popen(f"imodtrans -R 0 aunps/combined_savedpicks_asim.mod aunps/combined_savedpicks_rotxed.mod",shell=True)
        pro.wait()
        pro = Popen(f"imodtrans -I {tomo_name}_full_rec.mrc aunps/combined_savedpicks_rotxed.mod aunps/combined_savedpicks_rotxed_asrec.mod",shell=True)
        pro.wait()
        pro = Popen(f"rawtiltcoords -ro {tomo_name} -vo {tomo_name}_full_rec.mrc -ce aunps/combined_savedpicks_rotxed_asrec.mod -ou aunps_tiltseries.mod -re 0",shell=True)
        pro.wait()
        pro = Popen(f"ccderaser -ExpandCircleIterations 3 -InputFile {tomo_name}.mrc -OutputFile {tomo_name}_aunperase.mrc -ModelFile aunps_tiltseries.mod -BetterRadius 7.0 -PolynomialOrder ' -1' -CircleObjects / -MergePatches 1",shell=True)
        pro.wait()
        
    # Run 3dctfsetup
    tmp_path = "/tmp/3dctftemp"
    Path(tmp_path).mkdir(exist_ok=True)
    image_stack = Path(f"{'../' if tilt_parent else ''}{tomo_name}{'_aunperase' if delete_aunps else ''}{custom_suffix}.mrc")
    #with mrcfile.open(image_stack,mode="r+") as f:
    #    f.voxel_size = 1.0
    pro1 = Popen(f"ctf3dsetup -th 30 -f -red {bin_factor} {'-e' if remove_beads else ''} -ra {image_stack} {'-x '+str(custom_xf) if custom_xf else ''} -un -parallel -te {tmp_path} -rec tilt.com", stdout=PIPE, stderr=PIPE, shell=True)
    out, error = pro1.communicate()
    print(out)
    print(error)
    
    
    pro1 = Popen(f"xfmodel -back -scale 0.125 -xform {tomo_name}.xf {tomo_name}_erase.fid {tomo_name}_erase_raw.fid", stdout=PIPE, stderr=PIPE, shell=True)
    out, error = pro1.communicate()
    print(out)
    print(error)
    # Run 3dctf
    gpustrings = [f":{str(i+1)}" for i in range(num_gpus)]
    print(f"processchunks -G localhost{''.join(gpustrings)} ctf3d")
    pro2 = Popen(f"processchunks -G localhost{''.join(gpustrings)} ctf3d", shell=True)
    pro2.wait()

    # run clip rotx
    pro3 = Popen(f"clip rotx {tomo_name}_3dctf_rec.mrc {tomo_name}_full_rec_{'SIRT' if FakeSirtIterations > 0 else 'BP'}_3DCTF_BIN{bin_factor}{'_AUNPERASED' if delete_aunps else ''}{'CX' if custom_xf else ''}{custom_suffix}.mrc", shell=True)
    pro3.wait()

    
@app.command()
def write_dynamo_table(
    positions_set: str,
):
    import dynamotable
    import starfile
    try:
        tomo_name = list(Path.cwd().glob("*.rawtlt"))[0].stem
    except IndexError:
        typer.echo("No rawtlt file found")
        raise typer.Exit()
    rlnTomoImportParticleFile = Path.cwd() / "ampa_positions" / f"{positions_set}.star"
    relion_data = starfile.read(rlnTomoImportParticleFile)['particles']

    dynamo_dicts = [
        {
            "x": x["rlnCoordinateX"] * 2.0,
            "y": x["rlnCoordinateY"] * 2.0,
            "z": x["rlnCoordinateZ"] * 2.0,
            "tag": i+1,
            "tdrot": x["rlnAnglePsi"] + 90,
            "tilt": - x["rlnAngleTilt"],
            "narot": x["rlnAngleRot"] + 270,
        } for i, x in relion_data.iterrows()
    ]

    dynamo_data = pd.DataFrame(dynamo_dicts)
    dynamotable.write(dynamo_data, Path.cwd() / "ampa_positions" / f"{positions_set}.tbl")

@app.command()
def dynamo_table_to_relion4_star(
    dynamo_table: Path,
    tomo_doc: Path,
    particle_star_file: Path
):
    import dynamotable
    import starfile
    import pandas

    dynamo_data = dynamotable.read(dynamo_table)

    tomo_data = pd.read_csv(tomo_doc,delim_whitespace=True,header=None,names=['tomo','tomo_file'],dtype={'tomo':'int64'})

    combined = dynamo_data.merge(tomo_data,on='tomo',how='inner')
    combined['rlnCoordinateX'] = (combined['x'] + combined['dx']) * 2
    combined['rlnCoordinateY'] = (combined['y'] + combined['dy']) * 2
    combined['rlnCoordinateZ'] = (combined['z'] + combined['dz']) * 2
    combined['rlnAngleRot'] = combined['narot'] - 270
    combined['rlnAngleTilt'] = - combined['tilt']
    combined['rlnAnglePsi'] = combined['tdrot'] - 90
    combined['rlnTomoName'] = combined['tomo_file'].str.split('/').str[0]
    
    relion_df = combined[['rlnCoordinateX','rlnCoordinateY','rlnCoordinateZ','rlnAngleRot','rlnAngleTilt','rlnAnglePsi','rlnTomoName']]

    print(relion_df)


   

    starfile.write(relion_df, particle_star_file)



    combined['x'] *=4
    combined['y'] *=4
    combined['z'] *=4

    print(combined[combined['tomo'] == 1][['x','y','z','tdrot','tilt','narot']])


from enum import Enum

class Tomotype(str, Enum):
    BP = "BP"
    DDW = "DDW"


@app.command()
def match_aunps(
    tomotype: Tomotype = Tomotype.BP,
    high_pass: Optional[float] = None,
):
    import torch
    import mrcfile
    from findingampa.utils.tm3d import calc_3dtm, fit_gaussian_3d_list
    try:
        tomo_name = list(Path.cwd().glob("*.rawtlt"))[0].stem
    except IndexError:
        typer.echo("No rawtlt file found")
        raise typer.Exit()
 

    tomo_suffix = {
        Tomotype.BP: "_full_rec_BP_flat.mrc",
        Tomotype.DDW: "_full_rec_BP_3DCTF_BIN4_ddw.mrc"
    }

    locations_tomo, locations_score_volume, score_volume = calc_3dtm(Path(tomo_name+tomo_suffix[tomotype]), high_pass=high_pass)
    score_volume = torch.tensor(score_volume).cuda()
    fitted_params_list = fit_gaussian_3d_list(torch.tensor(score_volume), locations_score_volume.cuda(),12)
    offsets = torch.tensor([[f['x0'], f['y0'], f['z0']] for f in fitted_params_list])
    with mrcfile.open(tomo_name+tomo_suffix[tomotype], permissive=True) as f:
        tomo_tensor = torch.tensor(f.data)

    tomo_fitted_params_list = fit_gaussian_3d_list(tomo_tensor.cuda(), locations_tomo.int().cuda(), 8, sign=-1)
    data = pd.DataFrame({'faTM_PeakX':locations_tomo[:,2].cpu().numpy(),
                        'faTM_PeakY':locations_tomo[:,1].cpu().numpy(),
                        'faTM_PeakZ':locations_tomo[:,0].cpu().numpy(),
                        'faTM_PeakScore':[score_volume[x,y,z].item() for x,y,z in locations_score_volume],
                        'faTM_gaussianFitX':offsets[:,0].cpu().numpy(),
                        'faTM_gaussianFitY':offsets[:,1].cpu().numpy(),
                        'faTM_gaussianFitZ':offsets[:,2].cpu().numpy(),
                        'faTM_gaussianFitSigmaX':[f['sigma_x'] for f in fitted_params_list],
                        'faTM_gaussianFitSigmaY':[f['sigma_y'] for f in fitted_params_list],
                        'faTM_gaussianFitSigmaZ':[f['sigma_z'] for f in fitted_params_list],
                        'faTM_gaussianFitAmplitude':[f['amplitude'] for f in fitted_params_list],
                        'faTomo_gaussianFitX':[f['x0'] for f in tomo_fitted_params_list],
                        'faTomo_gaussianFitY':[f['y0'] for f in tomo_fitted_params_list],
                        'faTomo_gaussianFitZ':[f['z0'] for f in tomo_fitted_params_list],
                        'faTomo_gaussianFitSigmaX':[f['sigma_z'] for f in tomo_fitted_params_list],
                        'faTomo_gaussianFitSigmaY':[f['sigma_y'] for f in tomo_fitted_params_list],
                        'faTomo_gaussianFitSigmaZ':[f['sigma_x'] for f in tomo_fitted_params_list],
                        'faTomo_gaussianFitAmplitude':[f['amplitude'] for f in tomo_fitted_params_list]})
    aunppositions = Path.cwd() / "aunps" 
    aunppositions.mkdir(exist_ok=True)
    output_filename = aunppositions / f"aunp_tm_{tomotype.value}{'_HP' + str(high_pass) if high_pass is not None else ''}.star"
    starfile.write(data,output_filename)
    print(f"Wrote potential picks to {output_filename}")

@app.command()
def check_pixelsizes():
    import mrcfile
    try:
        tomo_name = list(Path.cwd().glob("*.rawtlt"))[0].stem
    except IndexError:
        typer.echo("No rawtlt file found")
        raise typer.Exit()
    
    to_test = [
        {"name": f"{tomo_name}.mrc", "expected": 2.5},
        {"name": f"../{tomo_name}.mrc", "expected": 2.5},
        {"name": f"../{tomo_name}_EVN.mrc", "expected": 2.5},
        {"name": f"../{tomo_name}_ODD.mrc", "expected": 2.5},
        {"name": f"{tomo_name}_EVN.mrc", "expected": 2.5},
        {"name": f"{tomo_name}_ODD.mrc", "expected": 2.5},
        {"name": f"{tomo_name}_ali.mrc", "expected": 10.0},
    ]

    passed = True

    for test in to_test:
        if not Path(test["name"]).exists():
            print(f"File {test['name']} not found")
            continue
        with mrcfile.open(test["name"], permissive=True) as f:
            if f.voxel_size.x != test["expected"]:
                print(f"Pixel size of {test['name']} is {f.voxel_size.x}, expected {test['expected']}")
                passed = False
    
    return passed

@app.command()
def correct_pixelsizes(
    force: bool = False
):
    import mrcfile
    from subprocess import Popen, PIPE
    from findingampa.utils.imod import check_ccderaser
    try:
        tomo_name = list(Path.cwd().glob("*.rawtlt"))[0].stem
    except IndexError:
        typer.echo("No rawtlt file found")
        raise typer.Exit()
    
    if check_pixelsizes() and not force:
        print("Pixel sizes already correct")
        return
    
    to_correct = [
        {"name": f"{tomo_name}.mrc", "expected": (2.5,2.5,1)},
        {"name": f"../{tomo_name}_EVN.mrc", "expected": (2.5,2.5,1)},
        {"name": f"../{tomo_name}_ODD.mrc", "expected": (2.5,2.5,1)},
        {"name": f"../{tomo_name}.mrc", "expected": (2.5,2.5,1)},
        {"name": f"{tomo_name}_EVN.mrc", "expected": (2.5,2.5,1)},
        {"name": f"{tomo_name}_ODD.mrc", "expected": (2.5,2.5,1)},
    ]
    

    #if check_ccderaser(tomo_name):
    #    Popen(f"submfg newst.com",shell=True).wait()
    for cor in to_correct:
        try:
            with mrcfile.open(cor["name"], mode="r+") as f:
                f.voxel_size = cor["expected"]
                
        except FileNotFoundError:
            print(f"File {cor['name']} not found")

    need_erasing = check_ccderaser(tomo_name)
    # Regenerate ali file
    Popen(f"submfg newst.com",shell=True).wait()

    if need_erasing:
        try:
            if not Path(f"{tomo_name}_erase_original.fid").exists():
                Path(f"{tomo_name}_erase.fid").rename(f"{tomo_name}_erase_original.fid")
        except FileNotFoundError:
            print("Error: erase fid not found")
        out, err = Popen(f"imodinfo {tomo_name}_erase_original.fid", shell=True, stdout=PIPE).communicate()
        for line in out.decode().split("\n"):
            if "#      SCALE" in line:
                scale = float(line.split()[-2][0])
        with mrcfile.open(f"{tomo_name}_ali.mrc", mode="r+") as f:
            xs,ys,zs = f.header["nx"], f.header["ny"], f.header["nz"]
        # Ok, need to (A) revert the scaling (B) scale the coordinates if collected from different binning (C) make sure it get the right image dimensions
        Popen(f"imodtrans -I {tomo_name}_ali.mrc -sx {scale/4.0} -sy {scale/4.0} -n {xs},{ys},{zs} {tomo_name}_erase_original.fid {tomo_name}_erase.fid",shell=True).wait()
        Popen(f"submfg golderaser.com",shell=True).wait()
        try:
            Path(f"{tomo_name}_erase_ali.mrc").rename(f"{tomo_name}_ali.mrc")
        except FileNotFoundError:
            print("Error: Gold erasing failed")
    mtffilter_com = Path.cwd() / "mtffilter.com"
    with open(mtffilter_com, "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "DoseWeightingFile" in line:
            lines[i] = f"DoseWeightingFile {tomo_name}.dose\n"
    with open(mtffilter_com, "w") as f:
        f.writelines(lines)
    Popen(f"submfg mtffilter.com",shell=True).wait()
    try:
        Path(f"{tomo_name}_filt_ali.mrc").rename(f"{tomo_name}_ali.mrc")
    except FileNotFoundError:
        print("Error: Dose filtering failed")
    
        # Need to run 
    


    if check_pixelsizes():
        print("Pixel sizes corrected")
    else:
        print("ERROR!: Pixel sizes not corrected")

@app.command()
def render_active_zonograms():
    import bpy
    import mrcfile
    import numpy as np
    try:
        tomo_name = list(Path.cwd().glob("*.rawtlt"))[0].stem
    except IndexError:
        typer.echo("No rawtlt file found")
        raise typer.Exit()
    
    active_zonogram_folder = Path("active_zonograms")
    active_zonogram_folder.mkdir(exist_ok=True)

    from findingampa.utils.analysis import generate_active_zonograms, render_active_zonograms, render_active_zone_extend_via_blender

    active_zonogram_data = generate_active_zonograms(tomo_name)

    print(f"{len(active_zonogram_data)} active zonograms generated")

    for i, az in enumerate(active_zonogram_data):
        mrcfile.write(active_zonogram_folder / f"active_zonogram_{i}.mrc", az[2].numpy(), overwrite=True)
        np.save(active_zonogram_folder / f"active_zonogram_{i}.npy",{"cs": az[0], "center": az[1], "objects": az[3]}, allow_pickle=True)

        fig = render_active_zonograms(az)

        fig.savefig(active_zonogram_folder / f"active_zonogram_{i}.png")

        render_active_zone_extend_via_blender(str(Path.cwd() / "aunps/new_aunp_template_CJS_ddw.blend"), az[1], np.array(az[2].shape)[::-1], az[0], active_zonogram_folder / f"active_zonogram_{i}_position.png")
    
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

    data_structure = pd.DataFrame(selected_aunp_pos, columns=["faCoordinateX", "faCoordinateY", "faCoordinateZ"])
    data_structure["active_zone"] = -1
    data_structure["postsynapse"] = False
    model = ImodModel()
    az_pick_filename = str(aunpfilename_filename.with_suffix(""))+f"_active_zone.mod"
    for i, az in enumerate(azs):
        data_structure.loc[az_picks[i], "active_zone"] = i
        az_id = str(az.with_suffix("")).split("_")[-1]
        filename = az_pick_filename[:-4] + f"_{az_id}.star"
        starfile.write(data_structure.loc[az_picks[i]], filename)
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
    filename = az_pick_filename[:-4] + f"_all.star"
    starfile.write(data_structure, filename)
    print(f"Saved picks to {filename}")

