from findingampa.project_management import FAContext, Tomogram
from findingampa.pipeline3.pipeline3_data import Pipeline3Data, Pipeline3TomogramResults, Pipeline3ClusterRegion
from pathlib import Path
from pydantic import BaseModel


def create_initial_reconstruction(ctx: FAContext, tomogram:Tomogram, output_path: Path):
    from findingampa.utils.reconstruct import reconstruct_by_relion
    tomo_resconstruction_folder = Path(output_path) / "reconstructions" / tomogram.FATomogramName
    tomo_resconstruction_folder.mkdir(parents=True,exist_ok=True)
    filename = tomo_resconstruction_folder / (tomogram.FATomogramName + "_initial.mrc")
    reconstruct_by_relion(ctx.obj.project.settings.FARelionTomoStar,tomogram.FATomogramName,filename,snr=100.0)
    return filename

def run_findbeads3d(ctx: FAContext, subregion: Pipeline3ClusterRegion, tomogram_results: Pipeline3TomogramResults, output_path: Path):
    from subprocess import run

    tomogram = next(tomogram for tomogram in ctx.obj.tomograms if tomogram.FATomogramName == subregion.pl3TomogramName)
    beads_model_file_dir = (output_path / "findbeads")
    beads_model_file_dir.mkdir(parents=True, exist_ok=True)
    beads_model_file = beads_model_file_dir / f"{subregion.pl3TomogramName}_{subregion.pl3CluserId}.mod"
    beads_point_file = beads_model_file_dir / f"{subregion.pl3TomogramName}_{subregion.pl3CluserId}.txt"
    tomogram_path = tomogram_results.pl3InitialReconstruction
    run(f"{ctx.obj.project.settings.FAImodPath}/findbeads3d -inp {tomogram_path} -ti {Path('tomograms')/tomogram.FATomogramName/(tomogram.FATomogramName + '.tlt')} -si 2.1 -thresh 20 -sto 0.5 -o {beads_model_file} -x {subregion.pl3Xmin},{subregion.pl3Xmax} -z {subregion.pl3Zmin},{subregion.pl3Zmax} -yminmax {subregion.pl3Ymin},{subregion.pl3Ymax}", shell=True)
    run(f"{ctx.obj.project.settings.FAImodPath}/model2point -ob -v -1 {beads_model_file} {beads_point_file}", shell=True)
    subregion.pl3AuNPModelfile= beads_model_file
