from findingampa.project_management import FAContext, Tomogram
from findingampa.pipeline1.pipeline1_data import Pipeline1Data, Pipeline1TomogramResults, Pipeline1ClusterRegion
from pathlib import Path

def update_modelfile(ctx: FAContext, data, tomogram):
    import pandas as pd
    import itertools
    import subprocess
    from pathlib import Path
    import tempfile
    # Generate temporary model file
    temp_dir = tempfile.mkdtemp()
    model_path = Path(temp_dir) / "c.mod"
    txt_path = Path(temp_dir) / "c.txt"
    
    model_items = []
    object_id = 0
    for subregion in data.pl1ClusterRegions.values():
        if subregion.pl1TomogramId == tomogram.FATomogramID:
            object_id += 1
            
            for y in range(subregion.pl1Ymin,subregion.pl1Ymax+1):
                model_items.append([object_id, y, subregion.pl1Xmin, y, subregion.pl1Zmin])
                model_items.append([object_id, y, subregion.pl1Xmax, y, subregion.pl1Zmin])
                model_items.append([object_id, y, subregion.pl1Xmax, y, subregion.pl1Zmax])
                model_items.append([object_id, y, subregion.pl1Xmin, y, subregion.pl1Zmax])
                
    
    df = pd.DataFrame(model_items, columns=["obj","con","x","y","z"])
    df.to_csv(txt_path, sep=" ", header=None, index=False, mode="a")
    subprocess.run(f"{ctx.obj.project.Settings.FAImodPath}/point2model {txt_path} {model_path}", shell=True)
    return model_path

def annotate_subregion(ctx, tomogram, data, data_file, maxclusterid):
    import mrcfile
    from subprocess import Popen, PIPE
    from time import sleep
    import typer

    with mrcfile.open(ctx.obj.project.Settings.FADataDir / tomogram.FABin10Tomogram) as mrc:
                ydim = mrc.header.ny

    # Launch 3dmod in a way that allows read/write to stdin/stdout

    p_3dmod  = Popen(f"{ctx.obj.project.Settings.FAImodPath}/3dmod -Y -L {ctx.obj.project.Settings.FADataDir / tomogram.FABin10Tomogram}", 
                    shell=True, 
                    #cwd=findbeads_path,
                    stdin=PIPE, 
                    stdout=PIPE, 
                    stderr=PIPE,
                    bufsize=1,
                    universal_newlines=True)
    sleep(1)
    while p_3dmod.poll() is None:
        sleep(0.1)
        modelfilename = update_modelfile(ctx, data, tomogram)
        p_3dmod.stdin.write(f"7 {modelfilename}\n")
        sleep(0.1)
        readnextline = True
        while readnextline and p_3dmod.poll() is None:
            line = p_3dmod.stderr.readline()
            if not modelfilename.exists():
                print(f"none {line}")
                line = p_3dmod.stderr.readline()
            print(f"Up {line}")
            if "OK" in line:
                readnextline = False
        # Wait for Upuser to confirm
        typer.confirm("Please annotate subregions in 3dmod or close 3dmod and press enter to continue",abort=False)
        # Send commands to 3dmod
        if p_3dmod.poll() is not None:
            break
        p_3dmod.stdin.write("10\n")
        sleep(0.1)
        readnextline = True
        found_rubberband = False
        
        while readnextline and p_3dmod.poll() is None:
            line = p_3dmod.stderr.readline()
            print("Down ", line)
            if "Rubberband" in line:
                found_rubberband = True
                print(f"Found {line}")
                values = line.split()
                if len(values) < 7:
                    typer.echo("Please also assign High and Low Z values")
                    break
                # Read values, taking roation around x into account
                _ , xmin, zmin, xmax, zmax, ymin, ymax = values
                #typer.echo(ctx, output_path, tomogram, xmin, ymin, xmax, ymax, zmin, zmax)
                ymin = ydim - int(ymin)
                ymax = ydim - int(ymax)
                ymin, ymax = ymax, ymin
                print(f"Adding subregion Xmin {xmin} Xmax {xmax} Ymin {ymin} Ymax {ymax} Zmin {zmin} Zmax {zmax}")
                data.pl1ClusterRegions[maxclusterid+1] = Pipeline1ClusterRegion(
                    pl1CluserId=maxclusterid+1,
                    pl1TomogramId=tomogram.FATomogramID,
                    pl1Xmin=xmin,
                    pl1Xmax=xmax,
                    pl1Ymin=ymin,
                    pl1Ymax=ymax,
                    pl1Zmin=zmin,
                    pl1Zmax=zmax,
                )
                maxclusterid += 1
                data.write(data_file)
            if "OK" in line:
                readnextline = False
                if not found_rubberband:
                    typer.echo("Please select a region using rubberband")
                    break


def annotate_threshold_and_synaptic_membranes(
        ctx: FAContext,
        tomogram: Tomogram,
        data: Pipeline1Data,
        data_file: Path,
        subregion: Pipeline1ClusterRegion,
):
    from findingampa.pipeline1.pipeline1_steps import create_model_file_for_subregion, write_aunp_model_file, write_synaptic_contour_model_file
    from subprocess import Popen, PIPE
    from rich.prompt import Prompt
    import pandas as pd

    output_path = ctx.obj.project.Settings.FADataDir / "pipeline_data" / "pipeline1" / "subregions" / f"{tomogram.FATomogramName}_{subregion.pl1CluserId}"
    output_path.mkdir(parents=True, exist_ok=True)\
    
    included_objects = create_model_file_for_subregion(ctx=ctx,
                                                        tomogram=tomogram,
                                                        tomogram_results=data.pl1TomogramResults[tomogram.FATomogramID],
                                                        subregion=subregion,
                                                        output_path=output_path)


    p_3dmod = Popen(f"{ctx.obj.project.Settings.FAImodPath}/3dmod -L {ctx.obj.project.Settings.FADataDir / tomogram.FABin10Tomogram} {output_path / 'temp_model.mod'}",
            shell=True, 
            #cwd=findbeads_path,
            stdin=PIPE, 
            stdout=PIPE, 
            stderr=PIPE,
            bufsize=1,
            universal_newlines=True)
    line = ""
    p_3dmod.stdin.write("8\n")
    while "OK" not in line:
        line = p_3dmod.stderr.readline()
    p_3dmod.stdin.write("3\n")
    line = ""
    while "OK" not in line:
        line = p_3dmod.stderr.readline()
    
    included_objects
    threshold = Prompt.ask("Please enter the desired threshold value", default=str(subregion.pl1AuNPThreshold))
    if subregion.pl1PresynapticObjectId is not None:
        try:
            pre_synaptic = included_objects.get_loc(subregion.pl1PresynapticObjectId) + 2
        except KeyError:
            pre_synaptic = None
    else:
        pre_synaptic = None
    pre_synaptic = int(Prompt.ask("Please enter the model number of the presynaptic membrane", default=str(pre_synaptic)))
    if subregion.pl1PostsynapticObjectId is not None:
        try:
            post_synaptic = included_objects.get_loc(subregion.pl1PostsynapticObjectId) + 2
        except KeyError:
            post_synaptic = None
    else:
        post_synaptic = None
    post_synaptic = int(Prompt.ask("Please enter the model number of the postsynaptic membrane", default=str(post_synaptic)))
        
    subregion.pl1AuNPThreshold = threshold
    subregion.pl1PresynapticObjectId = included_objects[pre_synaptic-2]
    subregion.pl1PostsynapticObjectId = included_objects[post_synaptic-2]
    p_3dmod.terminate()
    
    write_aunp_model_file(ctx=ctx,
                            AuNPModelfile=ctx.obj.project.Settings.FADataDir / subregion.pl1AuNPModelfile,
                            threshold=subregion.pl1AuNPThreshold,
                            output_path=output_path / "aunp.txt")
    subregion.pl1AuNPThresholdedModel = (output_path / "aunp.txt").relative_to(ctx.obj.project.Settings.FADataDir)
    write_synaptic_contour_model_file(ctx=ctx,
                                        ContourModelfile=ctx.obj.project.Settings.FADataDir / data.pl1TomogramResults[tomogram.FATomogramID].pl1SegmentationContourFile,
                                        object_id=subregion.pl1PresynapticObjectId,
                                        output_path=output_path / "presynaptic.txt")
    subregion.pl1PresynapticModel = (output_path / "presynaptic.txt").relative_to(ctx.obj.project.Settings.FADataDir)
    write_synaptic_contour_model_file(ctx=ctx,
                                        ContourModelfile=ctx.obj.project.Settings.FADataDir / data.pl1TomogramResults[tomogram.FATomogramID].pl1SegmentationContourFile,
                                        object_id=subregion.pl1PostsynapticObjectId,
                                        output_path=output_path / "postsynaptic.txt")
    subregion.pl1PostsynapticModel = (output_path / "postsynaptic.txt").relative_to(ctx.obj.project.Settings.FADataDir)
    data.write(data_file)