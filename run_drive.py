drive_loc = "/Volumes/AARYAN_PSSD/"

import glob
import os

filenames = list(sorted(glob.glob(drive_loc + "iris*")))
bad_results = []

for l in filenames:
    f = l.split("/")[-1][:-3]
    os.system(f"gzip -dc < {l} > ~/Documents/Code/LMSAL_HUB/iris_hub/{f}")
    os.system(f"python3 sunspot_kmeans.py {f}")
    os.remove(f)

    all_files = glob.glob(f"~/Documents/Code/LMSAL_HUB/iris_hub/{f[:-5]}/*")
    if "kmeans_data.jbl.gz" not in all_files:
        bad_results.append(f)

    if "k_map" not in all_files:
        pdf = "_".join(f.split("_")[:4] + f.split("_")[5:])[:-5]
        os.system(
            f"cp ~/Documents/Code/LMSAL_HUB/iris_hub/pdf/k_map_{pdf}_k160.pdf ~/Documents/Code/LMSAL_HUB/iris_hub/{f[:-5]}"
        )

    # if not os.path.isfile(
    #     f"~/Documents/Code/LMSAL_HUB/iris_hub/{f[:-5]}/kmeans_data.jbl.gz"
    # ):
    #     bad_results.append(f)


file = open("bad.txt", "w")
file.write("\n".join(bad_results))
file.close()
