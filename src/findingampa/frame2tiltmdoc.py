# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mdocfile",
#     "feud[all]",
# ]
# ///
import feud
from mdocfile.data_models import Mdoc, MdocGlobalData, MdocSectionData
from pathlib import Path
from typing import Optional

def create(
        base: str,
        dose_rate: float
) -> None:
    mdoc_files = Path.cwd().glob(f"{base}*.*.mdoc")
    print(mdoc_files)
    mdoc = Mdoc(
        titles=['[ Spoofed using mdocfile ]'],
        global_data=MdocGlobalData(),
        section_data=[],
    )
    for mdoc_file in mdoc_files:
        mdoc_frames = Mdoc.from_file(mdoc_file)
        section_data = MdocSectionData()
        data = mdoc_frames.section_data[0].model_dump()
        data.pop('ZValue')
        for k,v in data.items():
            if v is None:
                continue
            setattr(section_data, k, v)
        mdoc.section_data.append(section_data)
    mdoc.section_data = sorted(mdoc.section_data, key=lambda x: x.DateTime)
   
    for i,tilt in enumerate(mdoc.section_data):
        tilt.ExposureDose = dose_rate
        tilt.ZValue = i
    print(mdoc.to_dataframe())
    with open(f"{base}.mdoc", "w") as f:
        f.write(mdoc.to_string())

if __name__ == "__main__":
    feud.run(create)