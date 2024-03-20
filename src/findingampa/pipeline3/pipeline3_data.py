from pydantic import BaseModel, validator
from findingampa._typeutils import NoneOrNan
from pathlib import Path
import starfile
import pandas as pd


class Pipeline3Results(BaseModel):
    """Results from pipeline1."""
    pl3PosesStarFile: NoneOrNan[Path] = None
    pl3TomogramsStarFile: NoneOrNan[Path] = None
   


class Pipeline3TomogramResults(BaseModel):
    """Results from pipeline1."""
    pl3TomogramName: str
    pl3InitialReconstruction: NoneOrNan[Path]


class Pipeline3ClusterRegion(BaseModel):
    pl3CluserId: int
    pl3TomogramName: str
    pl3Xmin: int 
    pl3Xmax: int
    pl3Ymin: int
    pl3Ymax: int
    pl3Zmin: int
    pl3Zmax: int
    pl3AuNPModelfile: NoneOrNan[Path] = None
    pl3AuNPThreshold: NoneOrNan[float] = None
    pl3AuNPThresholdedModel: NoneOrNan[Path] = None
    

    class Config:
        validate_assignment = True
    


class Pipeline3Data(BaseModel):
    """Data to be used in pipeline1."""
    pl3Results: Pipeline3Results = Pipeline3Results()
    pl3TomogramResults: dict[str, Pipeline3TomogramResults] = {}
    pl3ClusterRegions: dict[int, Pipeline3ClusterRegion] = {}

    @classmethod
    def read(cls, filename: Path):
        data = starfile.read(filename, always_dict=True)

        if "pl3TomogramResults" not in data:
            data["pl3TomogramResults"] = pd.DataFrame()
        if "pl3ClusterRegions" not in data:
            data["pl3ClusterRegions"] = pd.DataFrame()
        return cls(
            pl3Results=Pipeline3Results(
                **data['pl3Results']
            ),
            pl3TomogramResults={row["pl3TomogramName"]: Pipeline3TomogramResults(**row) for row in data["pl3TomogramResults"].apply(lambda x: x.to_dict(),axis=1).to_list()},
            pl3ClusterRegions={row["pl3CluserId"]: Pipeline3ClusterRegion(**row) for row in data["pl3ClusterRegions"].apply(lambda x: x.to_dict(),axis=1).to_list()},
        )
    
    def write(self, filename: Path):

        out_dict = {
            "pl3Results": self.pl3Results.model_dump(),
        }
        if len(self.pl3TomogramResults) > 0:
            out_dict["pl3TomogramResults"] = pd.DataFrame([row.model_dump() for row in self.pl3TomogramResults.values()])
        if len(self.pl3ClusterRegions) > 0:    
            out_dict["pl3ClusterRegions"] = pd.DataFrame([row.model_dump() for row in self.pl3ClusterRegions.values()])
        
        starfile.write(out_dict, filename)