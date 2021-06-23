drive_loc = "/Volumes/AARYAN_PSSD/"

import glob
import os
from iris_lmsalpy import saveall as sv

filenames = list(sorted(glob.glob(drive_loc + "iris*")))
bad_results = []
all_masks = {}

for l in filenames:
    f = l.split("/")[-1][:-3]
    os.system(f"gzip -dc < {l} > ~/Documents/Code/LMSAL_HUB/iris_hub/{f}")
    os.system(f"python3 sunspot_kmeans.py {f}")
    os.remove(f)

    all_files = glob.glob(f"/Users/aaryan/Documents/Code/LMSAL_HUB/iris_hub/{f[:-5]}/*")
    has_kmeans = any("kmeans_data.jbl.gz" in i for i in all_files)
    if not has_kmeans:
        bad_results.append(f)

    has_pdf = any("k_map" in i for i in all_files)
    if not has_pdf:
        pdf = "_".join(f.split("_")[:4] + f.split("_")[5:])[:-5]
        os.system(
            f"cp ~/Documents/Code/LMSAL_HUB/iris_hub/pdf/k_map_{pdf}_k160.pdf ~/Documents/Code/LMSAL_HUB/iris_hub/{f[:-5]}"
        )

    aux = sv.load(f"{f[:-5]}/masked_image.jbl.gz")
    all_masks[f[:-5]] = aux["data_mask"]
    os.remove(f"{f[:-5]}/masked_image.jbl.gz")

with open("bad.txt", "w") as file:
    file.write("\n".join(bad_results))

sv.save("all_masked_images_v2.jbl.gz", all_masks, force=True)

