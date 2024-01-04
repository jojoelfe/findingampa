from pydantic import BaseModel, validator, BeforeValidator
from pathlib import Path
from findingampa._typeutils import NoneOrNan
from typing import Optional
import typer


class Settings(BaseModel):
    FADataDir: Path
    FAImodPath: str = "$IMOD_DIR/bin"
    FAMC2Path: str = "/groups/cryoadmin/software/MOTIONCOR2/motioncor2-1.4.2/MotionCor2_1.4.2_Cuda111-02-15-2020"
    FACistemPath: str = "/groups/cryoadmin/software/CISTEM/je_dev/"

class Tomogram(BaseModel):
    FATomogramID: int
    FATomogramName: str
    FAUnbinnedTomogram: Path
    FABin10Tomogram: Path
    FAMovies: NoneOrNan[Path] = None
    FAMoviesGain: NoneOrNan[Path] = None
    FAExposureDose: NoneOrNan[float] = None
    FAPL1SkipManualSteps: bool = False
    FAPL1ExcludeFromFinal: bool = False

class Project(BaseModel):
    Settings: Settings
    Data: list[Tomogram] = []

class FAGlobal(BaseModel):
    project: Project
    tomograms: list[Tomogram]
    project_path: Path

class FAContext(typer.Context):
    obj: Optional[FAGlobal] = None

class relionMicrograph(BaseModel):
    rlnFileName: Path
    
class relionParticle(BaseModel):
    rlnOriginalParticleName: Path
