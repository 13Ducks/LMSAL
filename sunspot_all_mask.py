from astropy.io.fits.util import first
from iris_lmsalpy import extract_irisL2data, saveall as sv
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os, shutil
import glob
import sunpy.map
from sunpy.net import Fido, attrs as a
import astropy.units as u
from astropy.coordinates import SkyCoord
from datetime import datetime, timedelta


def run_masking(raster_filename):
    dir_name = "all_images"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    iris_raster = extract_irisL2data.load(raster_filename, window_info=["Mg II k 2796"])

    mgii = iris_raster.raster["Mg II k 2796"]
    first_null = 0
    for i, t in enumerate(mgii.date_time_acq_ok):
        if t == "-- ::":
            first_null += 1
        else:
            break

    last_null = -1
    for i, t in enumerate(mgii.date_time_acq_ok[::-1]):
        if t == "-- ::":
            last_null -= 1
        else:
            break

    # downloaded_files = {
    #     "hmi": "/Users/aaryan/Documents/Code/LMSAL_HUB/iris_hub/all_images/other_data/hmi_m_45s_2019_04_09_04_58_30_tai_magnetogram.fits",
    #     "aia304": "/Users/aaryan/Documents/Code/LMSAL_HUB/iris_hub/all_images/other_data/aia_lev1_304a_2019_04_09t04_57_29_13z_image_lev1.fits",
    #     "aia1700": "/Users/aaryan/Documents/Code/LMSAL_HUB/iris_hub/all_images/other_data/aia_lev1_1700a_2019_04_09t04_57_40_73z_image_lev1.fits",
    #     "aia4500": "/Users/aaryan/Documents/Code/LMSAL_HUB/iris_hub/all_images/other_data/aia_lev1_4500a_2019_04_09t05_00_05_68z_image_lev1.fits",
    # }

    other_data = download_other_data(mgii, dir_name, first_null, last_null)
    climit = 1.5 * mgii.data.mean() + 5

    epsilon = 1e-2
    ps_wl = 2810.58

    find_wl_idx = lambda wl: np.where(np.abs(mgii.wl - wl) < epsilon)[0][0]
    ps_wl_idx = find_wl_idx(ps_wl)
    best = ps_wl_idx

    data = mgii.data

    bounds = [0, data.shape[0]]
    for r in range(data.shape[0]):
        if data[r].mean() > 0:
            bounds[0] = r
            break

    for r in range(data.shape[0] - 1, 0, -1):
        if data[r].mean() > 0:
            bounds[1] = r
            break

    bounds[0] += 1
    data = data[bounds[0] : bounds[1], :, best]
    orig_data = mgii.data[:, :, best]

    diffs = np.abs(np.diff(np.median(data, axis=1)))
    mdev = np.median(diffs)
    s = diffs / mdev if mdev else 0.0
    lines = np.where(s > 10)[0]

    black = np.where(data < 0)
    black_mask = np.ones(shape=data.shape, dtype=np.uint8)
    black_mask[black] = 0
    black_mask[lines] = 0

    data = cv2.medianBlur(data, 3)
    m = data[np.where(data >= 0)].mean()

    area_thresholds = [0.3, 0.75]

    mask_umbra = data < (m * area_thresholds[0])
    mask_penumbra = (data > (m * area_thresholds[0])) & (
        data < (m * area_thresholds[1])
    )
    mask_quiet = data > m * area_thresholds[1]
    masks = [mask_umbra, mask_penumbra, mask_quiet, mask_quiet]

    res_masks = []
    ellipse_mask = np.zeros(
        shape=[orig_data.shape[0], orig_data.shape[1]], dtype=np.float32
    )
    blank_sections = np.zeros(
        shape=[orig_data.shape[0], orig_data.shape[1]], dtype=np.float32
    )

    section_to_num = [1, 2, 3, 6]

    for i, m in enumerate(masks):
        w = np.where(m, data, 0)

        if res_masks:
            res = mask_layer(w, False, black_mask, res_masks[i - 2])
        else:
            res = mask_layer(w, i == 1, black_mask, None)

        mask, sp, quiet, ellipse = res
        blank_sections[bounds[0] : bounds[1], :][
            (mask > 0) & (blank_sections[bounds[0] : bounds[1], :] == 0)
        ] = section_to_num[i]

        if sp is not None:
            res_masks.extend([sp, quiet])
            ellipse_mask[bounds[0] : bounds[1], :] = ellipse

    full_data = np.zeros(
        shape=[orig_data.shape[0], orig_data.shape[1]], dtype=np.float32
    )
    full_data[bounds[0] : bounds[1], :] = data

    aux = orig_data * 0
    maxth = np.nanmedian(orig_data) * 1.25
    aux[orig_data > maxth] = 1

    blank_sections[((aux == 1) & (blank_sections == 3))] = 4
    blank_sections[((aux == 1) & (blank_sections == 6))] = 5

    full_data = np.zeros(
        shape=[mgii.data.shape[0], mgii.data.shape[1]], dtype=np.float32
    )
    full_data[bounds[0] : bounds[1], :] = data

    all_data = create_figure(
        mgii,
        full_data,
        climit,
        other_data,
        blank_sections,
        ellipse_mask,
        dir_name,
        first_null,
        last_null,
        shifted=False,
        filename="/" + raster_filename[:-5] + ".png",
    )

    create_figure(
        mgii,
        full_data,
        climit,
        other_data,
        blank_sections,
        ellipse_mask,
        dir_name,
        first_null,
        last_null,
        shifted=True,
        filename="/" + raster_filename[:-5] + "_shifted.png",
    )

    del iris_raster, mgii

    shutil.rmtree(f"{os.getcwd()}/{dir_name}/other_data/")
    return all_data


def find_files(begin, end, instrument, field, wl=None):
    # should add logic for closest
    begin = datetime.strptime(begin, "%Y-%m-%d %H:%M:%S")
    end = datetime.strptime(end, "%Y-%m-%d %H:%M:%S")

    if wl:
        res = Fido.search(a.Time(begin, end), instrument, field, wl)
    else:
        res = Fido.search(a.Time(begin, end), instrument, field)

    while res.file_num == 0:
        begin -= timedelta(hours=12)
        end += timedelta(hours=12)

        print(f"begin {begin} end {end}")

        if wl:
            res = Fido.search(a.Time(begin, end), instrument, field, wl)
        else:
            res = Fido.search(a.Time(begin, end), instrument, field)

    return res


def download_other_data(mgii, dir_name, first_null, last_null, downloaded=False):
    if not downloaded:
        tt = mgii.date_time_acq_ok

        begin = tt[first_null]
        end = tt[last_null]

        result_hmi = find_files(
            begin, end, a.Instrument.hmi, a.Physobs.los_magnetic_field
        )

        result_aia304 = find_files(
            begin,
            end,
            a.Instrument.aia,
            a.Physobs.intensity,
            a.Wavelength(304 * u.angstrom),
        )
        result_aia1700 = find_files(
            begin,
            end,
            a.Instrument.aia,
            a.Physobs.intensity,
            a.Wavelength(1700 * u.angstrom),
        )
        result_aia4500 = find_files(
            begin,
            end,
            a.Instrument.aia,
            a.Physobs.intensity,
            a.Wavelength(4500 * u.angstrom),
        )

        results = {
            "hmi": result_hmi,
            "aia304": result_aia304,
            "aia1700": result_aia1700,
            "aia4500": result_aia4500,
        }
    else:
        results = downloaded

    submaps = {}

    for k, res in results.items():
        print(k)
        if not downloaded:
            closest = (None, float("inf"))
            t = datetime.strptime(tt[len(tt) // 2], "%Y-%m-%d %H:%M:%S")
            for i, r in enumerate(res._list[0]):
                d = datetime.strptime(r["time"]["start"], "%Y%m%d%H%M%S")
                diff = abs((d - t).total_seconds())
                if diff < closest[1]:
                    closest = (i, diff)
            downloaded_file = Fido.fetch(
                res[0, closest[0]],
                path=f"{os.getcwd()}/{dir_name}/other_data/",
                max_conn=1,
            )
        else:
            downloaded_file = res

        full_map = sunpy.map.Map(downloaded_file)

        if k == "hmi":
            full_map = full_map.rotate(order=3)

        first_x = mgii["XCENIX"][first_null]
        last_x = mgii["XCENIX"][last_null]

        bottom_left = SkyCoord(
            first_x * u.arcsec,
            ((mgii["YCEN"] - mgii.extent_arcsec_arcsec[3] / 2)) * u.arcsec,
            frame=full_map.coordinate_frame,
        )
        top_right = SkyCoord(
            last_x * u.arcsec,
            ((mgii["YCEN"] + mgii.extent_arcsec_arcsec[3] / 2)) * u.arcsec,
            frame=full_map.coordinate_frame,
        )

        submap = full_map.submap(bottom_left, top_right=top_right)
        submaps[k] = submap

    return submaps


def mask_layer(w, calc_super, black_mask, m):
    ret, masked_image = cv2.threshold(w, 0, 255, cv2.THRESH_BINARY)

    masked_image = masked_image.astype(np.uint8)
    masked_image &= black_mask

    kernel = np.ones((3, 3), np.uint8)
    masked_image = cv2.morphologyEx(masked_image, cv2.MORPH_CLOSE, kernel, iterations=1)
    super_penumbra = None
    quiet = None
    ellipse_mask = None

    thresh = masked_image.astype(np.uint8)
    cnts, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[-2:]

    for cnt in cnts:
        hull = cv2.convexHull(cnt)
        cv2.drawContours(thresh, [hull], -1, 255, -1)

    thresh = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, np.ones((11, 3), np.uint8), iterations=1
    )

    masked_image &= thresh

    if len(cnts) == 0:
        return (
            np.zeros(masked_image.shape, dtype=np.float32),
            super_penumbra,
            quiet,
            ellipse_mask,
        )

    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

    if calc_super:
        cnts, hierarchy = cv2.findContours(
            masked_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )[-2:]
        ellipse_mask = np.zeros(thresh.shape).astype(np.uint8)
        contour_mask = np.zeros(thresh.shape).astype(np.uint8)
        for c in cnts:
            area = cv2.contourArea(cv2.convexHull(c))

            if (area / thresh.size) > 0.001:
                ellipse = cv2.fitEllipse(c)
                center, r, ang = ellipse
                new_ellipse = (center, (r[0] * 2.3, r[1] * 2.3), ang)

                # -1 thickness causes it to be filled in
                cv2.ellipse(ellipse_mask, new_ellipse, 255, -1)
                cv2.drawContours(contour_mask, [c], -1, 255, -1)

        super_penumbra = ~contour_mask & ellipse_mask
        quiet = ~ellipse_mask

    if m is not None:
        masked_image &= m

    masked_image &= black_mask
    masked_image = masked_image.astype(np.float32)

    return (masked_image, super_penumbra, quiet, ellipse_mask)


def create_figure(
    mgii,
    full_data,
    climit,
    other_data,
    blank_sections,
    sp_mask,
    dir_name,
    first_null,
    last_null,
    shifted=False,
    filename="/all_data.png",
):
    all_data = {
        "iris": full_data,
        "mask": blank_sections,
        "sp_mask": sp_mask,
        "mu": mgii.mu,
    }

    extent_heliox_helioy = mgii.extent_heliox_helioy
    extent_heliox_helioy[0] = mgii["XCENIX"][first_null]
    extent_heliox_helioy[1] = mgii["XCENIX"][last_null]

    where = np.where((blank_sections == 1) | (blank_sections == 2), blank_sections, 0)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    order = ["aia1700", "aia304", "hmi", "aia4500"]

    im = axes[0].imshow(
        full_data, cmap=mgii.cmap, extent=extent_heliox_helioy, origin="lower"
    )
    im.set_clim([0, climit])
    axes[0].set_title("iris")

    for i, k in enumerate(order):
        submap = other_data[k]
        all_data[k] = submap.data

        im = axes[i + 1].imshow(
            submap.data,
            cmap=submap.cmap,
            norm=submap.plot_settings["norm"],
            extent=extent_heliox_helioy,
            origin="lower",
        )
        axes[i + 1].set_title(k)

        extent_use = extent_heliox_helioy
        if shifted:
            dx = submap.center.Tx.value - mgii.XCEN
            dy = submap.center.Ty.value - mgii.YCEN

            extent_use = [
                extent_heliox_helioy[0] - dx,
                extent_heliox_helioy[1] - dx,
                extent_heliox_helioy[2] - dy,
                extent_heliox_helioy[3] - dy,
            ]

        color = "#ff0000" if k == "hmi" else "#ffffff"

        axes[i + 1].contour(
            where,
            levels=[1, 2],
            extent=extent_use,
            colors=[color],
        )

        if k == "hmi":
            axes[i + 1].contour(
                sp_mask,
                levels=[1],
                extent=extent_use,
                colors=["#ff0000"],
            )
            im.set_clim([-500, 500])

    axes[5].imshow(
        blank_sections, extent=extent_heliox_helioy, origin="lower", cmap="inferno"
    )
    axes[5].set_title("masks")

    plt.suptitle(filename[1:-4])
    fig.text(0.5, 0.04, "Helioprojective Longitude [arcsec]", ha="center", va="center")
    fig.text(
        0.06,
        0.5,
        "Helioprojective Latitude [arcsec]",
        ha="center",
        va="center",
        rotation="vertical",
    )

    plt.savefig(dir_name + filename)

    return all_data


if __name__ == "__main__":
    # test = run_masking("iris_l2_20131120_141151_3883006146_raster_t000_r00000.fits")

    # raise Exception("rip")
    drive_loc = "/Volumes/AARYAN_PSSD/"

    filenames = list(sorted(glob.glob(drive_loc + "iris*")))
    all_data = {}
    start = 29
    filenames = filenames[start:]

    curr = start

    loaded = sv.load("all_data.jbl.gz")
    if loaded:
        all_data = loaded["all_data"]

    for l in filenames:
        f = l.split("/")[-1][:-3]
        os.system(f"gzip -dc < {l} > ~/Documents/Code/LMSAL_HUB/iris_hub/{f}")
        try:
            data = run_masking(f)
        except:
            sv.save("all_data.jbl.gz", all_data, force=True)
            raise

        os.remove(f)
        all_data[f[:-5]] = data

        curr += 1
        print(curr)
        break

    sv.save("all_data.jbl.gz", all_data, force=True)
