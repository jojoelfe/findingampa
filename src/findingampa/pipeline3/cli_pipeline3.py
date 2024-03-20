import typer
from rich import print
from rich.panel import Panel
from rich.text import Text
app = typer.Typer()

from findingampa.pipeline3.pipeline3_data import Pipeline3Data, Pipeline3TomogramResults, Pipeline3ClusterRegion
from findingampa.project_management import FAContext, Tomogram
from pathlib import Path
import pandas as pd


@app.command()
def run_pipeline3(
    ctx: FAContext,
):
    """Run pipeline 3"""

    from pathlib import Path
    import mrcfile
    
    from findingampa.pipeline3.pipeline3_steps import create_initial_reconstruction, run_findbeads3d
    from findingampa.pipeline1.pipeline1_steps import (
                                    annotate_membrane, 
                                    convert_qpix_into_imod, 
                                    estimate_ampa_poses,
                                    prepare_aunp_and_synaptic_membrane_model,
                                    create_visualization_files
                                    )
    from findingampa.pipeline3.pipeline3_steps_interactive import (
                                    annotate_subregion,
                                    )
    
    


    output_path = Path("finding_ampa") / "pipeline_data" / "pipeline3"
    output_path.mkdir(parents=True, exist_ok=True)
    data_file = output_path / "pipeline3_data.star"
    if data_file.exists():
        data = Pipeline3Data.read(data_file)
    else:
        data = Pipeline3Data()
        data.write(data_file)
    

    message = Text("Starting pipeline3", style="bold red")
    panel = Panel(message)
    print(panel)

    # Annotate membranes

    print(Text("Creating initial SNR100 reconstructions", style="bold blue"))
    for tomogram in ctx.obj.tomograms:

        # Skip if tomogram already segmented
        if tomogram.FATomogramName in data.pl3TomogramResults and data.pl3TomogramResults[tomogram.FATomogramName].pl3InitialReconstruction is not None:
            print(Text(f"In {tomogram.FATomogramName} already done", style="green"))
            continue
        print(Text(f"{tomogram.FATomogramName}", style="green"))
        reconstruction_file = create_initial_reconstruction(ctx, tomogram, output_path)
        if tomogram.FATomogramName in data.pl3TomogramResults:
            data.pl3TomogramResults[tomogram.FATomogramName].pl3InitialReconstruction = reconstruction_file
        else:
            data.pl3TomogramResults[tomogram.FATomogramName] = Pipeline3TomogramResults(pl3TomogramName=tomogram.FATomogramName, pl3InitialReconstruction=reconstruction_file)
        data.write(data_file)
    
    print(Text("Annotating subregions with AuNP beads", style="bold blue"))

    maxclusterid = 0
    for tomogram in ctx.obj.tomograms:

        if tomogram.FAPL1SkipManualSteps:
            continue
        if len(data.pl3ClusterRegions) > 0:
            maxclusterid = max(data.pl3ClusterRegions.keys())
        print(Text(f"In {tomogram.FATomogramName}", style="green"))
        annotate_subregion(ctx, tomogram, data, data_file, maxclusterid)
    
    print(Text("Rund findbeads3d", style="bold blue"))

    for subregion in data.pl3ClusterRegions.values():
        if subregion.pl3AuNPModelfile is not None:
            print(Text(f"In {tomogram.FATomogramName} subregion {subregion.pl3CluserId} already done", style="green"))
            continue
        tomogram = next(tomogram for tomogram in ctx.obj.tomograms if tomogram.FATomogramName == subregion.pl3TomogramName)
        print(Text(f"In {tomogram.FATomogramName} subregion {subregion.pl3CluserId}", style="green"))
        run_findbeads3d(ctx, subregion, data.pl3TomogramResults[subregion.pl3TomogramName], output_path)
        data.write(data_file)

   