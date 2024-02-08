# /// script
# dependencies = ["mrcfile","numpy","yet_another_imod_wrapper","typer"]
# ///
from pathlib import Path

import mrcfile
import numpy as np

from yet_another_imod_wrapper import align_tilt_series_using_patch_tracking
import typer

def main(tilt_series: Path = typer.Argument(..., help="Path to the tilt series")):

    raw_tlt_file = tilt_series.with_suffix('.rawtlt')
    align_tilt_series_using_patch_tracking(
        tilt_series=mrcfile.read(tilt_series),
        tilt_angles=np.loadtxt(raw_tlt_file),
        pixel_size=2.5,
        nominal_rotation_angle=87,
        patch_size=220,
        patch_overlap_percentage=33,
        basename=tilt_series.stem,
        output_directory=str(tilt_series.parent)+'imod',
    )

if __name__ == "__main__":
    typer.run(main)
