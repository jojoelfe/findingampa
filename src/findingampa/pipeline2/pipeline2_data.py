from pydantic import BaseModel, validator
from findingampa._typeutils import NoneOrNan
from pathlib import Path
import starfile
import pandas as pd
import datetime
from typing import Optional

class Pipeline2Results(BaseModel):
    """Results from pipeline1."""
    pl2PosesStarFile: NoneOrNan[Path] = None
    pl2TomogramsStarFile: NoneOrNan[Path] = None
   
class Pipeline2TiltInfo(BaseModel):
    pl2MovieFilename: Path
    pl2Tilt: float
    pl2AcquisitionTime: datetime.datetime
    pl2MoviePixelSize: float
    pl2FrameNumber: int
    pl2AccumulationDose: NoneOrNan[float] = None
    pl2Defocus1: NoneOrNan[float] = None
    pl2Defocus2: NoneOrNan[float] = None
    pl2AstigmatismAngle: NoneOrNan[float] = None
    pl2CtfFitResolution: NoneOrNan[float] = None
    pl2TiltAxisAngle: NoneOrNan[float] = None
    pl2TiltAngle: NoneOrNan[float] = None
    

class Pipeline2TomogramResults(BaseModel):
    """Results from pipeline1."""
    pl2TomogramName: str
    pl2TiltInfo: Optional[list[Pipeline2TiltInfo]] = None
    pl2MotionCorrDone: bool = False
    pl2CtffindDone: bool = False
    pl2Stack: NoneOrNan[Path] = None
    pl2EvenStack: NoneOrNan[Path] = None
    pl2OddStack: NoneOrNan[Path] = None
    pl2Ctffindresults: NoneOrNan[Path] = None
    pl2DeconvolvedStack: NoneOrNan[Path] = None
    pl2DeconvolvedEvenStack: NoneOrNan[Path] = None
    pl2DeconvolvedOddStack: NoneOrNan[Path] = None
    pl2SegmentationFile: NoneOrNan[Path] = None
    pl2SegmentationContourFile: NoneOrNan[Path] = None

class Pipeline2ClusterRegion(BaseModel):
    pl2CluserId: int
    pl2TomogramName: str
    pl2Xmin: int 
    pl2Xmax: int
    pl2Ymin: int
    pl2Ymax: int
    pl2Zmin: int
    pl2Zmax: int
    pl2AuNPModelfile: NoneOrNan[Path] = None
    pl2AuNPThreshold: NoneOrNan[float] = None
    pl2AuNPThresholdedModel: NoneOrNan[Path] = None
    pl2PresynapticObjectId: NoneOrNan[int] = None
    pl2PresynapticModel: NoneOrNan[Path] = None
    pl2PostsynapticObjectId: NoneOrNan[int] = None
    pl2PostsynapticModel: NoneOrNan[Path] = None
    pl2AMPAPosesModel: NoneOrNan[Path] = None
    pl2AMPAPosesStarfile: NoneOrNan[Path] = None

    class Config:
        validate_assignment = True
    


class Pipeline2Data(BaseModel):
    """Data to be used in pipeline1."""
    pl2Results: Pipeline2Results = Pipeline2Results()
    pl2TomogramResults: dict[str, Pipeline2TomogramResults] = {}
    pl2ClusterRegions: dict[int, Pipeline2ClusterRegion] = {}

    @classmethod
    def read(cls, filename: Path):
        data = starfile.read(filename, always_dict=True)

        if "pl2TomogramResults" not in data:
            data["pl2TomogramResults"] = pd.DataFrame()
        else:
            data["pl2TomogramResults"]["pl2TiltInfo"] = data["pl2TomogramResults"]["pl2TomogramName"].apply(
                lambda x: [Pipeline2TiltInfo(**row) for row in starfile.read(filename.parent / "tiltseriesinfo" / (x + ".star")).apply(lambda x: x.to_dict(),axis=1).to_list()]
            )
        if "pl2ClusterRegions" not in data:
            data["pl2ClusterRegions"] = pd.DataFrame()
        
        
        return cls(
            pl2Results=Pipeline2Results(
                **data['pl2Results']
            ),
            pl2TomogramResults={row["pl2TomogramName"]: Pipeline2TomogramResults(**row) for row in data["pl2TomogramResults"].apply(lambda x: x.to_dict(),axis=1).to_list()},
            pl2ClusterRegions={row["pl2CluserId"]: Pipeline2ClusterRegion(**row) for row in data["pl2ClusterRegions"].apply(lambda x: x.to_dict(),axis=1).to_list()},
        )
    
    def write(self, filename: Path):

        out_dict = {
            "pl2Results": self.pl2Results.model_dump(),
        }
        if len(self.pl2TomogramResults) > 0:
            out_df = pd.DataFrame([row.model_dump() for row in self.pl2TomogramResults.values()])
            filenames = out_df['pl2TomogramName'].apply(lambda x: filename.parent / "tiltseriesinfo" / (x + ".star"))
            filenames.apply(lambda x: x.parent.mkdir(parents=True, exist_ok=True))
            filenames.apply(lambda x: starfile.write(pd.DataFrame([row.model_dump(mode='json') for row in self.pl2TomogramResults[x.stem].pl2TiltInfo]), x))
            out_df["pl2TiltInfo"] = filenames
            out_dict["pl2TomogramResults"] = out_df
        if len(self.pl2ClusterRegions) > 0:    
            out_dict["pl2ClusterRegions"] = pd.DataFrame([row.model_dump() for row in self.pl2ClusterRegions.values()])
        
        starfile.write(out_dict, filename)