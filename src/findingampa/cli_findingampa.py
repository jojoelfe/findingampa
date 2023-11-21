import typer
import starfile
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
from findingampa.cli_pipeline1 import app as pipeline1_app
from findingampa.project_management import Project, FAContext, FAGlobal

app = typer.Typer(
)

app.registered_commands += pipeline1_app.registered_commands

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
        project = Path.cwd() / "finding_ampa_project.star"
        if not project.exists():
            ctx.obj = SimpleNamespace(project = None, imod_path = imod_path)
            return
    project_object = starfile.read(project,always_dict=True)
    project_object["Data"] = project_object["Data"].apply(lambda x: x.to_dict(),axis=1).to_list()
    project_object = Project(**project_object)
    ctx.obj = FAGlobal(project = project_object, tomograms = project_object.Data, project_path = project)

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