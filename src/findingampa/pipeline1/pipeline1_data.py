from pydantic import BaseModel, validator
from findingampa._typeutils import NoneOrNan
from pathlib import Path
import starfile
import pandas as pd


class Pipeline1Results(BaseModel):
    """Results from pipeline1."""
    pl1PosesStarFile: NoneOrNan[Path] = None
    pl1TomogramsStarFile: NoneOrNan[Path] = None
   

class Pipeline1TomogramResults(BaseModel):
    """Results from pipeline1."""
    pl1TomogramId: int
    pl1SegmentationFile: NoneOrNan[Path] = None
    pl1SegmentationContourFile: NoneOrNan[Path] = None

class Pipeline1ClusterRegion(BaseModel):
    pl1CluserId: int
    pl1TomogramId: int
    pl1Xmin: int 
    pl1Xmax: int
    pl1Ymin: int
    pl1Ymax: int
    pl1Zmin: int
    pl1Zmax: int
    pl1AuNPModelfile: NoneOrNan[Path] = None
    pl1AuNPThreshold: NoneOrNan[float] = None
    pl1AuNPThresholdedModel: NoneOrNan[Path] = None
    pl1PresynapticObjectId: NoneOrNan[int] = None
    pl1PresynapticModel: NoneOrNan[Path] = None
    pl1PostsynapticObjectId: NoneOrNan[int] = None
    pl1PostsynapticModel: NoneOrNan[Path] = None
    pl1AMPAPosesModel: NoneOrNan[Path] = None
    pl1AMPAPosesStarfile: NoneOrNan[Path] = None

    class Config:
        validate_assignment = True
    


class Pipeline1Data(BaseModel):
    """Data to be used in pipeline1."""
    pl1Results: Pipeline1Results = Pipeline1Results()
    pl1TomogramResults: dict[int, Pipeline1TomogramResults] = {}
    pl1ClusterRegions: dict[int, Pipeline1ClusterRegion] = {}

    @classmethod
    def read(cls, filename: Path):
        data = starfile.read(filename, always_dict=True)

        if "pl1TomogramResults" not in data:
            data["pl1TomogramResults"] = pd.DataFrame()
        if "pl1ClusterRegions" not in data:
            data["pl1ClusterRegions"] = pd.DataFrame()
        return cls(
            pl1Results=Pipeline1Results(
                **data['pl1Results']
            ),
            pl1TomogramResults={row["pl1TomogramId"]: Pipeline1TomogramResults(**row) for row in data["pl1TomogramResults"].apply(lambda x: x.to_dict(),axis=1).to_list()},
            pl1ClusterRegions={row["pl1CluserId"]: Pipeline1ClusterRegion(**row) for row in data["pl1ClusterRegions"].apply(lambda x: x.to_dict(),axis=1).to_list()},
        )
    
    def write(self, filename: Path):

        out_dict = {
            "pl1Results": self.pl1Results.model_dump(),
        }
        if len(self.pl1TomogramResults) > 0:
            out_dict["pl1TomogramResults"] = pd.DataFrame([row.model_dump() for row in self.pl1TomogramResults.values()])
        if len(self.pl1ClusterRegions) > 0:    
            out_dict["pl1ClusterRegions"] = pd.DataFrame([row.model_dump() for row in self.pl1ClusterRegions.values()])
        
        starfile.write(out_dict, filename)