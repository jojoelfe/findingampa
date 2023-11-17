import typer
import starfile
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
from findingampa.cli_pipeline1 import app as pipeline1_app

app = typer.Typer(
)

app.registered_commands += pipeline1_app.registered_commands

@app.callback()
def main(
    ctx: typer.Context,
    project: Path = typer.Option(None, help="Path to wanted project file",rich_help_panel="Expert Options"),
    imod_path: Path = typer.Option("", help="Path to IMOD binaries"),
):
    """Finding AMPA processing pipeline"""
    if project is None:
        project = Path.cwd() / "finding_ampa_project.star"
        if not project.exists():
            ctx.obj = SimpleNamespace(project = None, imod_path = imod_path)
            return
    project_object = starfile.read(project,always_dict=True)
    project_object["project_path"] = project
    ctx.obj = SimpleNamespace(project = project_object, tomograms = project_object["Data"])

@app.command()
def new_project():
    """Create a new project file"""
    project = {
        "Settings" : {
            "FAImodPath" : "",
        },
        "Data" : pd.DataFrame(columns=["FATomogramID","FATomogramName","FAUnbinnedTomogram","FABin10Tomogram","FAMovies"]),
    }
    starfile.write(project,Path.cwd() / "finding_ampa_project.star")