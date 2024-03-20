from pydantic import BaseModel, validator, BeforeValidator
from pathlib import Path
from findingampa._typeutils import NoneOrNan
from typing import Optional
import typer


class Settings(BaseModel):
    FAImodPath: str = "$IMOD_DIR/bin"
    FAMC2Path: str = "/groups/cryoadmin/software/MOTIONCOR2/motioncor2-1.4.2/MotionCor2_1.4.2_Cuda111-02-15-2020"
    FACistemPath: str = "/groups/cryoadmin/software/CISTEM/je_dev/"
    FARelionTomoStar: Optional[Path] = None

class Tomogram(BaseModel):
    FATomogramName: str
    FAMovies: NoneOrNan[Path] = None
    FAMoviesGain: NoneOrNan[Path] = None
    FAPL1TomoSuffixForMembrain: str = "_full_rec.mrc"
    FAPL1TomoSuffixForFindbeads: str = "_full_rec.mrc"
    FAPL1SkipManualSteps: bool = False
    FAPL1ExcludeFromFinal: bool = False

class RelionTomogram(BaseModel):
    rlnTomoName: str
    rlnTomoTiltSeriesName: str
    rlnTomoFrameCount: int
    rlnTomoSizeX: int
    rlnTomoSizeY: int 
    rlnTomoSizeZ: int 
    rlnTomoHand: float 
    rlnOpticsGroupName: str
    rlnTomoTiltSeriesPixelSize: float 
    rlnVoltage: float 
    rlnSphericalAberration: float 
    rlnAmplitudeContrast: float 
    rlnTomoImportFractionalDose: float

class Project(BaseModel):
    settings: Settings = Settings()
    data: list[Tomogram] = []

    def write(self, filename: Path):
        import starfile
        import pandas as pd
        out_dict = {
            "settings": self.settings.model_dump(),
        }
        if len(self.data) > 0:
            out_dict["data"] = pd.DataFrame([row.model_dump() for row in self.data])
        
        starfile.write(out_dict, filename)

class FAGlobal(BaseModel):
    project: Project
    tomograms: list[Tomogram]
    project_path: Path
    project_directory: Path
    relion_tomograms: dict[str, RelionTomogram]

class FAContext(typer.Context):
    obj: FAGlobal 

class relionMicrograph(BaseModel):
    rlnFileName: Path
    
class relionParticle(BaseModel):
    rlnOriginalParticleName: Path
