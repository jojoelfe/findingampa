from typing import Optional, Annotated, Any, TypeVar
from math import isnan
from pydantic import BaseModel, validator, BeforeValidator
from pathlib import Path

def coerce_nan_to_none(x: Any) -> Any:
    if x is None:
        return x
    if type(x) is str:
        if x == 'None':
            return None
        return x
    if type(x) is Path:
        if x == 'None':
            return None
        return x
    if isnan(x):
        return None
    
    return x

T = TypeVar('T')

NoneOrNan = Annotated[Optional[T], BeforeValidator(coerce_nan_to_none)]