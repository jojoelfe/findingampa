from pathlib import Path
from subprocess import run

for ddw_path in Path.cwd().glob("*/best_alignment/*ddw.mrc"):
    # Get the path name
    tomo_path = Path(ddw_path).parent
    png_path = tomo_path / "pngs"
    png_path.mkdir(exist_ok=True)
    # Run the command
    #         mrc2tif -p -z 187,187 -a 0,0 ../${tn}/best_alignment/${tn}_full_rec_BP_3DCTF_BIN4_ddw.mrc ${tn}_center.png
    for slice in [120,150,180]:
        run(
            [
                "mrc2tif",
                "-p",
                "-z",
                f"{slice},{slice}",
                "-a",
                "0,0",
                f"{ddw_path}",
                f"{png_path}/{tomo_path.parent.name}_center",
            ]
        )

