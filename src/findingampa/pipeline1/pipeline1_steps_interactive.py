from findingampa.project_management import FAContext, Tomogram
from findingampa.pipeline1.pipeline1_data import Pipeline1Data, Pipeline1TomogramResults, Pipeline1ClusterRegion
from pathlib import Path

def update_modelfile(ctx: FAContext, data: Pipeline1Data, tomogram: Tomogram):
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
        if subregion.pl1TomogramName == tomogram.FATomogramName:
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

def annotate_subregion(ctx: FAContext, tomogram: Tomogram, data, data_file, maxclusterid):
    import mrcfile
    from subprocess import Popen, PIPE
    from time import sleep
    import typer
    import os

    tomogram_path = ctx.obj.project_directory.parent / (ctx.obj.relion_tomograms[tomogram.FATomogramName].rlnTomoTiltSeriesName[:-4] + tomogram.FAPL1TomoSuffixForFindbeads)
    with mrcfile.open(tomogram_path) as mrc:
                ydim = mrc.header.ny

    # Launch 3dmod in a way that allows read/write to stdin/stdout

    p_3dmod  = Popen(f"{ctx.obj.project.Settings.FAImodPath}/3dmod -Y -L {tomogram_path}", 
                    shell=True, 
                    #cwd=findbeads_path,
                    stdin=PIPE, 
                    stdout=PIPE, 
                    stderr=PIPE,
                    bufsize=1,
                    universal_newlines=True)
    os.set_blocking(p_3dmod.stderr.fileno(), False)
    sleep(1)
    while p_3dmod.poll() is None:
        sleep(0.1)
        modelfilename = update_modelfile(ctx, data, tomogram)
        p_3dmod.stdin.write(f"7 {modelfilename}\n")
        sleep(1.0)
        lines = p_3dmod.stderr.readlines()
        print(f"Startip {lines}")
        # Wait for Upuser to confirm
        typer.confirm("Please annotate subregions in 3dmod or close 3dmod and press enter to continue",abort=False)
        # Send commands to 3dmod
        if p_3dmod.poll() is not None:
            break
        p_3dmod.stdin.write("10\n")
        
        sleep(1.0)
        response_lines = p_3dmod.stderr.readlines()
        found_rubberband = False
        print(f"Response {response_lines}")
        for line in response_lines:
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
                    pl1TomogramName=tomogram.FATomogramName,
                    pl1Xmin=xmin,
                    pl1Xmax=xmax,
                    pl1Ymin=ymin,
                    pl1Ymax=ymax,
                    pl1Zmin=zmin,
                    pl1Zmax=zmax,
                )
                maxclusterid += 1
                data.write(data_file)
                break
        if not found_rubberband:
            typer.echo("No rubberband found, please try again")
        


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

    output_path = output_path / "subregions" / f"{tomogram.FATomogramName}_{subregion.pl1CluserId}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    included_objects = create_model_file_for_subregion(ctx=ctx,
                                                        tomogram=tomogram,
                                                        tomogram_results=data.pl1TomogramResults[tomogram.FATomogramName],
                                                        subregion=subregion,
                                                        output_path=output_path)

    tomogram_path = ctx.obj.project_directory.parent / (ctx.obj.relion_tomograms[tomogram.FATomogramName].rlnTomoTiltSeriesName[:-4] + tomogram.FAPL1TomoSuffixForFindbeads)
    p_3dmod = Popen(f"{ctx.obj.project.Settings.FAImodPath}/3dmod -L {tomogram_path} {output_path / 'temp_model.mod'}",
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
    data.write(data_file)
    
    