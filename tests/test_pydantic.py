from pydantic import BaseModel
from pathlib import Path
from findingampa._typeutils import NoneOrNan


class Test(BaseModel):
    path: NoneOrNan[Path] = None


test = Test()

print(test.path)
print(test.path.parent)