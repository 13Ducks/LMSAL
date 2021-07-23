# -*- coding: utf-8 -*-
# Author: Aaryan Agrawal <aaryanagrawal03@gmail.com>

""" Routines to mask sunspots and display them alongside other data sources of the same event. """

from datetime import datetime, timedelta
from sunpy.net import Fido, attrs as a
import astropy.units as u
import sunpy.map
from astropy.coordinates import SkyCoord
from iris_lmsalpy import extract_irisL2data
import numpy as np
import cv2
from math import ceil
import matplotlib.pyplot as plt
import warnings
import os, shutil

warnings.filterwarnings("ignore")


def download_sdo_data(mgii, path, instrument, wl=None, max_time_diff=360, crop=True):
    """
    Downloads and crops a specified file from SDO.
    ----------
    Parameters
    raster: MgII raster obtained from IRIS
    path: File path to store downloaded files
    instrument: Either AIA or HMI
    wl: Wavelength for use in AIA queries
    max_time_diff: Maximum number of minutes allowed between SDO and IRIS data
    crop: Whether or not to crop map with IRIS coordinates
    """

    instrument = instrument.lower()

    if instrument not in ["aia", "hmi"]:
        raise ValueError("Instrument must be either AIA or HMI")

    if instrument == "aia" and not wl:
        raise ValueError("AIA wavelength must be provided")

    if instrument == "hmi" and wl:
        print("The wavelength provided for a HMI file will be ignored.")

    tt = mgii.date_time_acq_ok
    mid = tt[len(tt) // 2]

    # If the middle of the raster is null, the extent, which relies on YCEN, cannot be resolved correctly
    if mid == "-- ::":
        print("There is no data at the middle of the raster. Nothing has been done.")
        return

    mid = datetime.strptime(mid, "%Y-%m-%d %H:%M:%S")
    begin = mid - timedelta(minutes=max_time_diff)
    end = mid + timedelta(minutes=max_time_diff)

    if instrument == "aia":
        res = Fido.search(
            a.Time(begin, end),
            a.Instrument.aia,
            a.Physobs.intensity,
            a.Wavelength(wl * u.angstrom),
        )
    else:
        res = Fido.search(
            a.Time(begin, end),
            a.Instrument.hmi,
            a.Physobs.los_magnetic_field,
        )

    if res.file_num == 0:
        print("No data was found in the specified time range.")
        return

    # Find the file which was taken at the closest time to the middle timestamp of raster
    closest = (None, float("inf"))

    for i, r in enumerate(res._list[0]):
        d = datetime.strptime(r["time"]["start"], "%Y%m%d%H%M%S")
        diff = abs((d - mid).total_seconds())
        if diff < closest[1]:
            closest = (i, diff)

    downloaded_file = Fido.fetch(
        res[0, closest[0]],
        path=path,
        max_conn=1,
    )

    full_map = sunpy.map.Map(downloaded_file)

    # HMI is mounted upside down so has to be rotated
    if instrument == "hmi":
        full_map = full_map.rotate(order=3)

    if crop:
        # find find non-null value at both start and end of raster
        first_null = next(
            i for i, t in enumerate(mgii.date_time_acq_ok) if t != "-- ::"
        )
        last_null = next(
            i for i, t in enumerate(mgii.date_time_acq_ok[::-1]) if t != "-- ::"
        )
        last_null = -(last_null + 1)

        # Extent in x direction is first non-null value at start and end
        # Extent in y direction is center plus/minus height of raster
        bottom_left = SkyCoord(
            mgii.XCENIX[first_null] * u.arcsec,
            ((mgii.YCEN - mgii.extent_arcsec_arcsec[3] / 2)) * u.arcsec,
            frame=full_map.coordinate_frame,
        )
        top_right = SkyCoord(
            mgii.XCENIX[last_null] * u.arcsec,
            ((mgii.YCEN + mgii.extent_arcsec_arcsec[3] / 2)) * u.arcsec,
            frame=full_map.coordinate_frame,
        )

        submap = full_map.submap(bottom_left, top_right=top_right)
        return submap
    else:
        return full_map


def create_sunspot_mask(mgii, return_sp=False):
    """
    Creates a sunspot mask of the umbra, penumbra, superpenumbra, and plage.
    ----------
    Parameters
    mgii: MgII raster obtained from IRIS
    return_sp: Whether or not to return the superpenumbra ellipse mask (for plotting)
    """

    def mask_layer(masked_image, calc_super, black_mask, m):
        masked_image = masked_image.astype(np.uint8)

        # Remove all null data found in image, needed as sunspot mask uses a less than which would include these locations
        masked_image &= black_mask

        # MORPH_CLOSE = dilation then erosion, fills in small holes
        masked_image = cv2.morphologyEx(
            masked_image, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
        )

        super_penumbra = quiet = ellipse_mask = None

        thresh = masked_image.astype(np.uint8)
        cnts, hierarchy = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[-2:]

        if len(cnts) == 0:
            return (
                np.zeros(masked_image.shape, dtype=np.float32),
                super_penumbra,
                quiet,
                ellipse_mask,
            )

        # Draw convex hull (convex polygon approximation) for each contour
        # Then use MORPH_OPEN, which is erosion then dilation, removes small noise
        # Reason for convex hull first is that penumbra would be eroded on each side, removing it
        # With this, able to preserve penumbra and remove noise
        for cnt in cnts:
            hull = cv2.convexHull(cnt)
            cv2.drawContours(thresh, [hull], -1, 255, -1)

        # Kernel of 15 x 3 as IRIS rasters are much taller than wide
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((15, 3), np.uint8))
        masked_image &= thresh

        if calc_super:
            # Find contours again with cleaned image
            # CHAIN_APPROX_NONE used so that fitEllipse is able to work
            cnts, hierarchy = cv2.findContours(
                masked_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )[-2:]
            ellipse_mask = np.zeros(thresh.shape).astype(np.uint8)
            contour_mask = np.zeros(thresh.shape).astype(np.uint8)

            for c in cnts:
                area = cv2.contourArea(cv2.convexHull(c))

                # Don't try to generate superpenumbras for tiny contours since fitEllipse errors if not enough points
                if (area / thresh.size) > 0.001:
                    ellipse = cv2.fitEllipse(c)
                    center, r, ang = ellipse

                    # Superpenumbra radius approximated to be 2.3x the radius of the penumbra
                    new_ellipse = (center, (r[0] * 2.3, r[1] * 2.3), ang)

                    cv2.ellipse(ellipse_mask, new_ellipse, 255, -1)
                    cv2.drawContours(contour_mask, [c], -1, 255, -1)

            super_penumbra = ~contour_mask & ellipse_mask
            quiet = ~ellipse_mask

        if m is not None:
            masked_image &= m

        masked_image &= black_mask
        masked_image = masked_image.astype(np.float32)

        return (masked_image, super_penumbra, quiet, ellipse_mask)

    # Photospheric wavelength that provides nice view of sunspot
    ps_wl = 2810.58
    best = np.argmin(np.abs(mgii.wl - ps_wl))

    # Detect black bars at top and bottom of data and remove them
    bounds = [0, mgii.data.shape[0]]
    for r in range(mgii.data.shape[0]):
        if mgii.data[r].mean() > 0:
            bounds[0] = r + 1
            break

    for r in range(mgii.data.shape[0] - 1, 0, -1):
        if mgii.data[r].mean() > 0:
            bounds[1] = r
            break

    data = mgii.data[bounds[0] : bounds[1], :, best]
    orig_data = mgii.data[:, :, best]

    # Detect fiducial lines using standard deviation of horizontal averages
    diffs = np.abs(np.diff(np.median(data, axis=1)))
    mdev = np.median(diffs)
    s = diffs / mdev if mdev else 0.0
    lines = np.where(s > 10)[0]

    # Add fiducial lines and other null data to negative mask
    black = np.where(data < 0)
    black_mask = np.ones(shape=data.shape, dtype=np.uint8)
    black_mask[black] = 0
    black_mask[lines] = 0

    # Blur data for more consistent and uniform results
    data = cv2.medianBlur(data, 3)
    m = data[np.where(data >= 0)].mean()

    # Percentage thresholds corresponding to umbra/penumbra and penumbra/quiet
    area_thresholds = [0.3, 0.75]

    mask_umbra = data < (m * area_thresholds[0])
    mask_penumbra = (data > (m * area_thresholds[0])) & (
        data < (m * area_thresholds[1])
    )
    mask_quiet = data > m * area_thresholds[1]
    masks = [mask_umbra, mask_penumbra, mask_quiet, mask_quiet]

    res_masks = []

    mask_sections = np.zeros(
        shape=[orig_data.shape[0], orig_data.shape[1]], dtype=np.float32
    )
    ellipse_mask = np.zeros(
        shape=[orig_data.shape[0], orig_data.shape[1]], dtype=np.float32
    )

    # 4 and 5 left out for plage in penumbra and quiet respectively
    section_to_num = [1, 2, 3, 6]

    for i, m in enumerate(masks):
        thresh = np.where(m, 1, 0)

        if res_masks:
            res = mask_layer(thresh, False, black_mask, res_masks[i - 2])
        else:
            res = mask_layer(thresh, i == 1, black_mask, None)

        mask, sp, quiet, ellipse = res
        mask_sections[bounds[0] : bounds[1], :][
            (mask > 0) & (mask_sections[bounds[0] : bounds[1], :] == 0)
        ] = section_to_num[i]

        if sp is not None:
            res_masks.extend([sp, quiet])
            ellipse_mask[bounds[0] : bounds[1], :] = ellipse

    # Get high values for plage
    maxth = np.nanmedian(orig_data) * 1.25

    mask_sections[((orig_data > maxth) & (mask_sections == 3))] = 4
    mask_sections[((orig_data > maxth) & (mask_sections == 6))] = 5

    if return_sp:
        return (mask_sections, ellipse_mask)

    return mask_sections


def create_full_figure(
    mgii,
    sdo_data,
    mask_sections,
    save_path=None,
    shifted=False,
    draw_contours=True,
    sp_mask=None,
):
    """
    Create a figure with graphs of IRIS and SDO data.
    ----------
    Parameters
    mgii: MgII raster obtained from IRIS
    sdo_data: List of submaps obtained from SDO
    mask_sections: Layer mask of the sunspot
    save_path: Path to save created image
    shifted: Whether to try to align centers of SDO and IRIS data
    draw_contours: Whether or not to draw contours of mask on SDO plots
    sp_mask: Superpenumbra ellipse to be drawn on HMI graphs
    """

    climit = 1.5 * mgii.data.mean() + 5

    # Photospheric wavelength that provides nice view of sunspot
    ps_wl = 2810.58
    best = np.argmin(np.abs(mgii.wl - ps_wl))
    data = mgii.data[:, :, best]

    # Find find non-null value at both start and end of raster
    first_null = next(i for i, t in enumerate(mgii.date_time_acq_ok) if t != "-- ::")
    last_null = next(
        i for i, t in enumerate(mgii.date_time_acq_ok[::-1]) if t != "-- ::"
    )
    last_null = -(last_null + 1)

    # Extent given is incorrect as uses first and last values without accounting for null values
    extent_heliox_helioy = mgii.extent_heliox_helioy
    extent_heliox_helioy[0] = mgii.XCENIX[first_null]
    extent_heliox_helioy[1] = mgii.XCENIX[last_null]

    # Only the umbra and penumbra contours are drawn
    where = np.where((mask_sections == 1) | (mask_sections == 2), mask_sections, 0)

    # Create figure with least number of rows possible
    rows = ceil((2 + len(sdo_data)) / 3)
    fig, axes = plt.subplots(rows, 3, figsize=(12, 2 + 3 * rows))
    axes = axes.flatten()

    im = axes[0].imshow(
        data, cmap=mgii.cmap, extent=extent_heliox_helioy, origin="lower"
    )
    im.set_clim([0, climit])

    tt = mgii.date_time_acq_ok
    mid = tt[len(tt) // 2]
    axes[0].set_title(f"IRIS | {mid}")

    axes[-1].imshow(
        mask_sections, extent=extent_heliox_helioy, origin="lower", cmap="inferno"
    )
    axes[-1].set_title("Masks")

    # Create global x-axis and y-axis texts
    plt.suptitle(mgii.date_in_filename)
    fig.text(0.5, 0.06, "Helioprojective Longitude [arcsec]", ha="center", va="center")
    fig.text(
        0.08,
        0.5,
        "Helioprojective Latitude [arcsec]",
        ha="center",
        va="center",
        rotation="vertical",
    )

    # Plot every submap from SDO's data with it's respective colormap and plot settings
    for i, submap in enumerate(sdo_data):
        if submap:
            im = axes[i + 1].imshow(
                submap.data,
                cmap=submap.cmap,
                norm=submap.plot_settings["norm"],
                extent=extent_heliox_helioy,
                origin="lower",
            )

            if submap.detector == "HMI":
                axes[i + 1].set_title(
                    f"{submap.detector} | {submap.date.datetime.strftime('%Y-%m-%d %H:%M:%S')}"
                )
            else:
                axes[i + 1].set_title(
                    f"{submap.detector} {int(submap.wavelength.value)} | {submap.date.datetime.strftime('%Y-%m-%d %H:%M:%S')}"
                )

            # Can try to match centers of IRIS and SDO data for a more accurate comparison
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

            color = "#ff0000" if submap.detector == "HMI" else "#ffffff"

            if draw_contours:
                axes[i + 1].contour(
                    where,
                    levels=[1, 2],
                    extent=extent_use,
                    colors=[color],
                )

            if submap.detector == "HMI":
                if sp_mask is not None and draw_contours:
                    axes[i + 1].contour(
                        sp_mask,
                        levels=[1],
                        extent=extent_use,
                        colors=["#ff0000"],
                    )
                im.set_clim([-500, 500])

    if save_path:
        plt.savefig(save_path)

    plt.show()


def mask_and_graph_sunspot(
    filename, download_path, image_save_path=None, delete_after=False, shifted=False
):
    """
    Integrated function that downloads select SDO data, creates a mask, and generates a figure.
    ----------
    Parameters
    filename: IRIS fits file path location
    download_path: Folder where to store downloaded SDO data
    image_save_path: File where to store created figure image
    """
    iris_raster = extract_irisL2data.load(
        filename,
        window_info=["Mg II k 2796"],
    )
    mgii = iris_raster.raster["Mg II k 2796"]

    aia304 = download_sdo_data(mgii, download_path, "AIA", wl=304)
    aia1700 = download_sdo_data(mgii, download_path, "AIA", wl=1700)
    aia4500 = download_sdo_data(mgii, download_path, "AIA", wl=4500)
    hmi = download_sdo_data(mgii, download_path, "HMI")

    b, sp = create_sunspot_mask(mgii, return_sp=True)
    create_full_figure(
        mgii,
        [aia1700, aia304, hmi, aia4500],
        b,
        save_path=image_save_path,
        shifted=shifted,
        sp_mask=sp,
    )

    del iris_raster, mgii

    if delete_after:
        shutil.rmtree(download_path)
        os.remove(filename)


if __name__ == "__main__":
    mask_and_graph_sunspot(
        filename="iris_l2_20151107_055951_3893010094_raster_t000_r00000.fits",
        download_path="/Users/aaryan/Documents/Code/LMSAL_HUB/iris_hub/test",
        image_save_path="/Users/aaryan/Documents/Code/LMSAL_HUB/iris_hub/test/image.png",
    )
