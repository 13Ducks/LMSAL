from iris_lmsalpy import extract_irisL2data, saveall as sv
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os, shutil
import sunpy.map
from sunpy.net import Fido, attrs as a
import astropy.units as u
from astropy.coordinates import SkyCoord
from datetime import datetime
from scipy.ndimage import zoom


def run_masking(raster_filename):
    dir_name = raster_filename[:-5]
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    iris_raster = extract_irisL2data.load(raster_filename, window_info=["Mg II k 2796"])

    mgii = iris_raster.raster["Mg II k 2796"]

    other_data = download_other_data(mgii, dir_name)
    climit = 1.5 * mgii.data.mean() + 5

    epsilon = 1e-2
    first_wl = 2794.00276664
    last_wl = 2806.01988627
    ps_wl = 2810.58

    find_wl_idx = lambda wl: np.where(np.abs(mgii.wl - wl) < epsilon)[0][0]

    first_wl_idx = find_wl_idx(first_wl)
    last_wl_idx = find_wl_idx(last_wl)
    ps_wl_idx = find_wl_idx(ps_wl)
    data_bounds = (first_wl_idx, last_wl_idx + 1)
    best = ps_wl_idx

    data = np.flip(mgii.data, axis=0)

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

        mask, sp, quiet = res
        blank_sections[bounds[0] : bounds[1], :][
            (mask > 0) & (blank_sections[bounds[0] : bounds[1], :] == 0)
        ] = section_to_num[i]

        if sp is not None:
            res_masks.extend([sp, quiet])

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
        mgii, full_data, climit, other_data, blank_sections, dir_name
    )

    del mgii

    shutil.rmtree(f"{os.getcwd()}/{dir_name}/other_data/")
    return all_data


def download_other_data(mgii, dir_name):
    tt = mgii.date_time_acq_ok

    result_hmi = Fido.search(
        a.Time(tt[0], tt[-1]), a.Instrument.hmi, a.Physobs.los_magnetic_field
    )
    result_aia304 = Fido.search(
        a.Time(tt[0], tt[-1]),
        a.Instrument.aia,
        a.Physobs.intensity,
        a.Wavelength(304 * u.angstrom),
    )
    result_aia1700 = Fido.search(
        a.Time(tt[0], tt[-1]),
        a.Instrument.aia,
        a.Physobs.intensity,
        a.Wavelength(1700 * u.angstrom),
    )
    result_aia4500 = Fido.search(
        a.Time(tt[0], tt[-1]),
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

    submaps = {}

    for k, res in results.items():
        print(k)
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
        )

        full_map = sunpy.map.Map(downloaded_file)

        if k == "hmi":
            full_map = full_map.rotate(order=3)

        bottom_left = SkyCoord(
            (mgii["XCENIX"][0]) * u.arcsec,
            ((mgii["YCEN"] - mgii.extent_arcsec_arcsec[3] / 2)) * u.arcsec,
            frame=full_map.coordinate_frame,
        )
        top_right = SkyCoord(
            (mgii["XCENIX"][-1]) * u.arcsec,
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
        return (np.zeros(masked_image.shape, dtype=np.float32), super_penumbra, quiet)

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

    return (masked_image, super_penumbra, quiet)


def draw_contours_data(blank_sections, sections, data, color):
    data_draw = data.data.copy()
    for i in sections:
        flipped = np.flip(np.where(blank_sections == i, 1, 0), axis=0)
        flipped = zoom(
            flipped,
            (
                data_draw.shape[0] / flipped.shape[0],
                data_draw.shape[1] / flipped.shape[1],
            ),
        )
        flipped = flipped.astype(np.uint8)
        cnts, hierarchy = cv2.findContours(
            flipped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )[-2:]
        cv2.drawContours(data_draw, cnts, -1, color, 1)

    return data_draw


def create_figure(mgii, full_data, climit, other_data, blank_sections, dir_name):
    all_data = {"iris": full_data, "mask": blank_sections, "mu": mgii.mu}

    aia4500_draw = draw_contours_data(
        blank_sections, [1, 2], other_data["aia4500"], (0, 0, 0)
    )
    hmi_draw = draw_contours_data(
        blank_sections, [3, 5], other_data["hmi"], (255, 255, 255)
    )

    to_draw = {"hmi": hmi_draw, "aia4500": aia4500_draw}

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    order = ["aia1700", "aia304", "hmi", "aia4500"]

    im = axes[0].imshow(
        full_data,
        cmap=mgii.cmap,
        extent=mgii.extent_arcsec_arcsec,
        origin="lower",
        interpolation="nearest",
    )
    im.set_clim([0, climit])
    axes[0].set_title("iris")

    for i, k in enumerate(order):
        submap = other_data[k]
        all_data[k] = submap.data

        axes[i + 1].imshow(
            submap.data,
            cmap=submap.cmap,
            norm=submap.plot_settings["norm"],
            interpolation="nearest",
        )
        axes[i + 1].set_title(k)

        if k in to_draw:
            axes[i + 1].imshow(
                to_draw[k],
                cmap=submap.cmap,
                norm=submap.plot_settings["norm"],
                interpolation="nearest",
            )

    axes[5].imshow(
        blank_sections,
        extent=mgii.extent_arcsec_arcsec,
        origin="lower",
        interpolation="nearest",
    )
    axes[5].set_title("masks")

    plt.suptitle(dir_name)
    plt.savefig(dir_name + "/all_data.png")

    return all_data


if __name__ == "__main__":
    a, mu = run_masking("iris_l2_20190409_044820_3893010094_raster_t000_r00000.fits")
    print(a)
    print(mu)
