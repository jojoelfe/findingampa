import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 
import typer
import starfile
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
from findingampa.pipeline1.cli_pipeline1 import app as pipeline1_app
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
def update_project(
    ctx: FAContext,
):
    ctx.obj.project.Data = pd.DataFrame([x.model_dump() for x in ctx.obj.tomograms])
    starfile.write(ctx.obj.project.model_dump(),Path.cwd() / "finding_ampa_project.star")

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

@app.command()
def test_centerline(
    contour_file: Path = typer.Argument(...,help="Path to contour file")
):
    import imodmodel
    from shapely.geometry import Polygon
    from centerline.geometry import Centerline
    import matplotlib.pyplot as plt
    data = imodmodel.read(contour_file)
    # Split data by contour_id and iterate over each contour
    for contour_id, contour_data in data.groupby("contour_id"):
        if contour_id < 17630:
            continue
        plt.figure()
        # Get the contour coordinates
        contour = contour_data[["x","y"]].to_numpy()
        # Get the centerline coordinates
        polygon = Polygon(contour)
        attributes = {"id": 1, "name": "polygon", "valid": True}
        centerline = Centerline(polygon, interpolation_distance=2.0, **attributes)
        for a in centerline.geometry.geoms:
            for b in a.coords:
                print(b)
        plt.plot(*polygon.exterior.xy)
        plt.show()

@app.command()
def test_centerline2(
    contour_file: Path = typer.Argument(...,help="Path to contour file"),
    qpix_data_filename: Path = typer.Argument(...,help="Path to qpix data file"),
    the_paramter: float = typer.Argument(0.5,help="The parameter")
):
    import imodmodel
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import linemerge
    from centerline.geometry import Centerline
    import matplotlib.pyplot as plt
    data = imodmodel.read(contour_file)
    import mrcfile

    with mrcfile.open(qpix_data_filename) as mrc:
        qpixdata = mrc.data.copy()
    qpixdata = qpixdata - qpixdata.min()
    new_contour_data = []
    # Split data by contour_id and iterate over each contour
    cont_id = 1
    data["z"] = data["z"].round(0).astype(int)
    for z, z_data in data.groupby("z"):

        #plt.figure()
        # Get the contour coordinates
        polygon_list = []
        for contour_id, contour_data in z_data.groupby("contour_id"):
            contour = contour_data[["x","y"]].to_numpy()
            if len(contour) < 3:
                continue
            polygon = Polygon(contour)
            polygon_list.append(Polygon(contour))
        
        # Get the centerline coordinates
        multipolygon = MultiPolygon(polygon_list)

        #for a in multipolygon.geoms:
        #    plt.plot(*a.exterior.xy)
        #plt.show()
        attributes = {"id": 1, "name": "polygon", "valid": True}
        centerline = Centerline(multipolygon, interpolation_distance=the_paramter, **attributes)
        # Only use centerline segments that are in the anotatedf membrane
        segments = [a for a in centerline.geometry.geoms if qpixdata[int(z),int(a.centroid.y),int(a.centroid.x)] > 0.5]
        # J
        cont = linemerge(segments)
        long_enough = [a for a in cont.geoms if a.length > 10.0]
        cont = linemerge(long_enough)
        for a in cont.geoms:
            for b in a.coords:
                new_contour_data.append([1, cont_id, b[0], b[1], z])
            cont_id += 1
    pd.DataFrame(new_contour_data).to_csv("temp_model.txt", sep=" ", header=None, index=False, mode="w")
            #plt.plot(*a.xy)


@app.command()
def test_skeletonize(
    qpix_data: Path = typer.Argument(...,help="Path to qpix data file")
):
    import mrcfile

    with mrcfile.open(qpix_data) as mrc:
        data = mrc.data.copy()
    data = data - data.min()
    print(data.min(),data.max())
    from skimage.morphology import skeletonize

    for z in range(data.shape[0]):
        skeleton = skeletonize(data[z,:,:])
        data[z,:,:] = skeleton
    with mrcfile.new(qpix_data.with_stem(qpix_data.stem + '_ske').with_suffix('.mrc'), overwrite=True) as mrc:
        mrc.set_data(data)