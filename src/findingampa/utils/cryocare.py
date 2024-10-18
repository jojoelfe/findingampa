

def apply_cryocare(tomo_name):
    import json
    from pathlib import Path
    from subprocess import Popen, PIPE

    from findingampa.utils.assets import FA_ASSETS

    predit_config = {
    "path": str(FA_ASSETS / "cryocare" / "top32_20240910.tar.gz"),
    "even": f"{tomo_name}_full_rec_BP_3DCTF_BIN4_EVN.mrc",
    "odd": f"{tomo_name}_full_rec_BP_3DCTF_BIN4_ODD.mrc",
    "n_tiles": [1,1,1],
    "output": "cryocare_denoised",
    "overwrite": True,
    "gpu_id": 0
    }
    with open("cryocare_predict.json", "w") as f:
        json.dump(predit_config, f)
    pro1 = Popen(f"/groups/elferich/.conda/envs/cryocare_11/bin/cryoCARE_predict.py --conf cryocare_predict.json", stdout=PIPE, stderr=PIPE, shell=True)
    out, error = pro1.communicate()
    print(out)
    print(error)
    output_file = Path("cryocare_denoised") / f"{tomo_name}_full_rec_BP_3DCTF_BIN4_EVN.mrc"
    # Move file to the correct location
    result_file = output_file.replace(f"{tomo_name}_full_rec_BP_3DCTF_BIN4_denoised.mrc")
    return result_file