import starfile
import pandas as pd

results_relion = []
for i in range(20):
    for j in range(20):
        results_relion.append({
                        "rlnCoordinateX": 0,
                        "rlnCoordinateY": i* 100,
                        "rlnCoordinateZ": j * 100,
                        "rlnOriginXAngst": 0,
                        "rlnOriginYAngst": 0,
                        "rlnOriginZAngst": 0,
                        "rlnAngleRot": i * 10,
                        "rlnAngleTilt": j * 10,
                        "rlnAnglePsi": 0,
                        "rlnImagePixelSize": 1.0,
                        "rlnMicrographName": "micrograph.mrc",
                        "rlnOpticsGroup": 1,
                        "faVectorX": 0,
                        "faVectorY": 0,
                        "faVectorZ": 0,
                    })
starfile.write({
                'particles': pd.DataFrame(results_relion), 
                'optics': pd.DataFrame([
                    {'rlnOpticsGroup':1}
                    ])
            },  "relion_test.star")