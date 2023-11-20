from pydantic import BaseModel, validator, BeforeValidator
from pathlib import Path
from findingampa._typeutils import NoneOrNan





class Settings(BaseModel):
    FADataDir: Path
    FAImodPath: str = "$IMOD_DIR/bin"

class Tomogram(BaseModel):
    FATomogramID: int
    FATomogramName: str
    FAUnbinnedTomogram: Path
    FABin10Tomogram: Path
    FAMovies: Path | NoneOrNan = ...

class Project(BaseModel):
    Settings: Settings
    Data: list[Tomogram] = []