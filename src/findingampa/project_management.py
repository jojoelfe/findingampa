from pydantic import BaseModel, validator, BeforeValidator
from pathlib import Path
from findingampa._typeutils import NoneOrNan
from typing import Optional
import typer


class Settings(BaseModel):
    FADataDir: Path
    FAImodPath: str = "$IMOD_DIR/bin"

class Tomogram(BaseModel):
    FATomogramID: int
    FATomogramName: str
    FAUnbinnedTomogram: Path
    FABin10Tomogram: Path
    FAMovies: NoneOrNan[Path] = None

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
