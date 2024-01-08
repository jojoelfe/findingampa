import typer
from rich import print
from rich.panel import Panel
from rich.text import Text
app = typer.Typer()

from findingampa.pipeline1.pipeline1_data import Pipeline1Data, Pipeline1TomogramResults, Pipeline1ClusterRegion
from findingampa.project_management import FAContext, Tomogram
from pathlib import Path
import pandas as pd


@app.command()
def run_pipeline1(
    ctx: FAContext,
):
    """Run pipeline 1"""

    from pathlib import Path
    import mrcfile
    
    from findingampa.pipeline1.pipeline1_steps import (
                                    annotate_membrane, 
                                    convert_qpix_into_imod, 
                                    run_findbeads3d,
                                    estimate_ampa_poses,
                                    prepare_aunp_and_synaptic_membrane_model,
                                    create_visualization_files
                                    )
    from findingampa.pipeline1.pipeline1_steps_interactive import (
                                    annotate_subregion,
                                    annotate_threshold_and_synaptic_membranes
                                    )
    
    # Step1: Annotate beads
    # In each tomogram subregions need to be annoated using xminxmax yminymax zminzmax (User rubberbadn in 3dmod)
    # ALso each subregion need an optimized threshold value
    # findbeads3d command is: 
    # findbeads3d -inp $bin10tomo  -c c.mod -si 1.5 -a -50,50 -yl -thresh 30 -sto $threshold -o aunp_$i.mod -x $xminxmax -z zminzmax -yminmax yminymax

    # Step2: Annotate membranes
    # Command is:
    # membrain segment --tomogram-path $binnedtomo --ckpt-path static/MemBrain_seg_v10_alpha.ckpt

    # Step3: Sort and split surfaces
    # Command is:
    # imodauto -h 1.1 #membrainsegmentation  membrane_contours.mod
    # imodsortsurf -s  membrane_contours.mod membrane_surfaces.mod
    # Identify postsynaptic membranes in 3dmod and write down numbers
    # Extract using:
    # imodextract $number membrane_surfaces.mod postsynatpic_membrane_$i.mod
    
    # Step4: Estimate AMPA poses
    # Needs as input a dictionary describing which AuNP patch is on which postsynaptic membrane
    #files_to_analyze = [
    #    [data_folder / "aunp1.mod", data_folder / "postsynaptic2.mod"],
    #    [data_folder / "aunp2.mod", data_folder / "postsynaptic3.mod"],
    #    [data_folder / "aunp3.mod", data_folder / "postsynaptic1.mod"],
    #    [data_folder / "aunp4.mod", data_folder / "postsynaptic1.mod"],
    #]
    # run code currently in find_ampa_poses.py

    # Two interactive steps: 

    # Assign subareas
    # Iterate over subareas, adjust threshold and confirm or set post/pre assignment
    output_path = ctx.obj.project.Settings.FADataDir / "pipeline_data" / "pipeline1"
    output_path.mkdir(parents=True, exist_ok=True)
    data_file = output_path / "pipeline1_data.star"
    if data_file.exists():
        data = Pipeline1Data.read(data_file)
    else:
        data = Pipeline1Data()
        data.write(data_file)
    

    message = Text("Starting pipeline1", style="bold red")
    panel = Panel(message)
    print(panel)

    # Annotate membranes

    print(Text("Annotating membranes", style="bold blue"))
    for tomogram in ctx.obj.tomograms:

        # Skip if tomogram already segmented
        if tomogram.FATomogramID in data.pl1TomogramResults and data.pl1TomogramResults[tomogram.FATomogramID].pl1SegmentationFile is not None:
            print(Text(f"In {tomogram.FATomogramName} already done", style="green"))
            continue
        print(Text(f"In {tomogram.FATomogramName}", style="green"))
        segmentation_file = annotate_membrane(ctx, tomogram, output_path)
        segmentation_file = segmentation_file.relative_to(ctx.obj.project.Settings.FADataDir)
        if tomogram.FATomogramID in data.pl1TomogramResults:
            data.pl1TomogramResults[tomogram.FATomogramID].pl1SegmentationFile = segmentation_file
        else:
            data.pl1TomogramResults[tomogram.FATomogramID] = Pipeline1TomogramResults(pl1TomogramId=tomogram.FATomogramID, pl1SegmentationFile=segmentation_file)
        data.write(data_file)

    # Convert segmentation to contours and split into objects
    print(Text("Converting membranes to imod modelfiles", style="bold blue"))
    for tomogram in ctx.obj.tomograms:
        if tomogram.FATomogramID in data.pl1TomogramResults and data.pl1TomogramResults[tomogram.FATomogramID].pl1SegmentationContourFile is not None:
            print(Text(f"In {tomogram.FATomogramName} already done", style="green"))
            continue
        print(Text(f"In {tomogram.FATomogramName}", style="green"))
        membrain_surfaces_file_point = convert_qpix_into_imod(ctx, tomogram, output_path, ctx.obj.project.Settings.FADataDir / data.pl1TomogramResults[tomogram.FATomogramID].pl1SegmentationFile)
        membrain_surfaces_file_point = membrain_surfaces_file_point.relative_to(ctx.obj.project.Settings.FADataDir)
        data.pl1TomogramResults[tomogram.FATomogramID].pl1SegmentationContourFile = membrain_surfaces_file_point
        data.write(data_file)


    # Annotate subregions
    print(Text("Annotating subregions with AuNP beads", style="bold blue"))

    maxclusterid = 0
    for tomogram in ctx.obj.tomograms:

        if tomogram.FAPL1SkipManualSteps:
            continue
        if len(data.pl1ClusterRegions) > 0:
            maxclusterid = max(data.pl1ClusterRegions.keys())
        print(Text(f"In {tomogram.FATomogramName}", style="green"))
        annotate_subregion(ctx, tomogram, data, data_file, maxclusterid)
            
    
                    
    # Run findbeads3d
    print(Text("Running findbeads3d", style="bold blue"))

    for subregion in data.pl1ClusterRegions.values():
        if subregion.pl1AuNPModelfile is not None:
            print(Text(f"In {tomogram.FATomogramName} subregion {subregion.pl1CluserId} already done", style="green"))
            continue
        tomogram = next(tomogram for tomogram in ctx.obj.tomograms if tomogram.FATomogramID == subregion.pl1TomogramId)
        print(Text(f"In {tomogram.FATomogramName} subregion {subregion.pl1CluserId}", style="green"))
        run_findbeads3d(ctx, subregion, output_path)
        data.write(data_file)
    
    # Assign threshold and pre- and post-synaptic membranes
    print(Text("Selecting threshold and synaptic membranes", style="bold blue"))
    
    for subregion in data.pl1ClusterRegions.values():
        tomogram = next(tomogram for tomogram in ctx.obj.tomograms if tomogram.FATomogramID == subregion.pl1TomogramId)
        if tomogram.FAPL1SkipManualSteps:
            continue
        print(Text(f"In {tomogram.FATomogramName} subregion {subregion.pl1CluserId}", style="green"))

        annotate_threshold_and_synaptic_membranes(ctx, tomogram, data, data_file, subregion)
        

        #run(f"{ctx.obj.project.Settings.FAImodPath}/3dmod -L -m  -x {subregion.pl1Xmin},{subregion.pl1Xmax} -z {subregion.pl1Zmin},{subregion.pl1Zmax} -y {subregion.pl1Ymin},{subregion.pl1Ymax} {ctx.obj.project.Settings.FADataDir / tomogram.FABin10Tomogram} {ctx.obj.project.Settings.FADataDir / 'temp_model.mod'}", shell=True)

    # Prepare AuNP and Synaptic Membrane Model
    print(Text("Preparing AuNP and synaptic membrane model", style="bold blue"))
    for subregion in data.pl1ClusterRegions.values():
        tomogram = next(tomogram for tomogram in ctx.obj.tomograms if tomogram.FATomogramID == subregion.pl1TomogramId)
        print(Text(f"In {tomogram.FATomogramName} subregion {subregion.pl1CluserId}", style="green"))
        prepare_aunp_and_synaptic_membrane_model(ctx, subregion,tomogram, data)
        data.write(data_file)
    
    # Define potential AMPA poses
    print(Text("Defining potential AMPA poses", style="bold blue"))
    for subregion in data.pl1ClusterRegions.values():
        if subregion.pl1AuNPThresholdedModel is not None and subregion.pl1PresynapticModel is not None and subregion.pl1PostsynapticModel is not None:
            tomogram = next(tomogram for tomogram in ctx.obj.tomograms if tomogram.FATomogramID == subregion.pl1TomogramId)
            print(Text(f"In {tomogram.FATomogramName} subregion {subregion.pl1CluserId}", style="green"))
            estimate_ampa_poses(ctx, subregion)
            data.write(data_file)
    
    # Create files for visualization in blender
    print(Text("Creating files for visualization in blender", style="bold blue"))
    for tomogram in ctx.obj.tomograms:
        print(Text(f"In {tomogram.FATomogramName}", style="green"))
        open_command = create_visualization_files(ctx, tomogram, data)
        print(Text(f"Open using 3dmod", style="yellow"))
        print(open_command)

    #aunp_statistic_rows = []
    #aunp_pair_statistic_rows = []
    #relion_file_row = []
    #for subregion in data.pl1ClusterRegions.values():
    #    statistic_rows.extend(calculate_statistics(ctx, subregion))
    #    relion_file_row.extend(create_relion_file(ctx, subregion))
    
    #write_overall_statisitcs_and_relion_file(ctx, statistic_rows, relion_file_row)
    
    

@app.command()
def visualize_3dmod(
        ctx: FAContext,
):
    import subprocess
    data = Pipeline1Data.read(ctx.obj.project.Settings.FADataDir / "pipeline_data" / "pipeline1" / "pipeline1_data.star")
    for tomogram in ctx.obj.tomograms:
        continue

# fOR BLENDER:

# iMODMESH CONTOURS
# imod2vrml2 *.wrl file
# Import to blender. Cancel rotations, scale by 0.1
# IMport relion poses
# Still need to figure out what to do about gold