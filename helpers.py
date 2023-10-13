import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from l2a_analysis import L2A_Analysis, L2A_Band, L2A_Band_Stack

class_dict = {
    "no_data": {
        "value": 0,
        "color": (0, 0, 0),
        "description": "no data",
    },
    "saturated_or_defective": {
        "value": 1,
        "color": (255, 0, 0),
        "description": "saturated or defective",
    },
    "casted_shadows": {
        "value": 2,
        "color": (50, 50, 50),
        "description": "casted shadows",
    },
    "cloud_shadows": {
        "value": 3,
        "color": (165, 42, 42),
        "description": "cloud shadows",
    },
    "vegetation": {
        "value": 4,
        "color": (0, 150, 0),
        "description": "vegetation",
    },
    "bare_soils": {
        "value": 5,
        "color": (255, 255, 0),
        "description": "bare soils",
    },
    "water": {
        "value": 6,
        "color": (0, 0, 255),
        "description": "water",
    },
    "unclassified": {
        "value": 7,
        "color": (128, 128, 128),
        "description": "unclassified",
    },
    "cloud_medium_probability": {
        "value": 8,
        "color": (211, 211, 211),
        "description": "cloud medium probability",
    },
    "cloud_high_probability": {
        "value": 9,
        "color": (255, 255, 255),
        "description": "cloud high probability",
    },
    "thin_cirrus": {
        "value": 10,
        "color": (0, 255, 255),
        "description": "thin cirrus",
    },
    "snow": {
        "value": 11,
        "color": (255, 192, 203),
        "description": "snow",
    },
}


def plot_rgb_image(
    product, red_band, green_band, blue_band, clip_value=65536, gamma=1, title=None
):
    if title is None:
        title = f"False colour RGB with bands {red_band}, {green_band}, {blue_band}"
    bands = product.read_bands([red_band, green_band, blue_band])
    rgb = np.dstack(bands)

    # clip values
    rgb[rgb > clip_value] = clip_value
    # normalize rgb
    rgb = rgb / rgb.max()

    # do gamma transformation
    rgb_gamma = np.power(rgb, 1 / gamma)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(
        rgb,
        interpolation="none",
    )
    ax.set_title(title)
    ax.set_axis_off()
    plt.show()


def plot_true_color_image(product, gamma=1, title="True Color Image", clip_value=65536):
    plot_rgb_image(product, "B04", "B03", "B02", clip_value, gamma, title)


def plot_band(
    product,
    band,
    color_map="Blues",
):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(
        product[band].read(1),
        interpolation="none",
        cmap=color_map,
    )
    ax.set_title(f"{band}")
    ax.set_axis_off()
    plt.show()


def band_difference_plot(reference, modified, band, color_map="Blues"):
    reference_array = reference[band].read(1).astype(np.float32)
    modified_array = modified[band].read(1).astype(np.float32)

    difference_array = reference_array - modified_array

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(
        difference_array,
        interpolation="none",
        cmap=color_map,
    )
    ax.set_title(f"{band}")
    ax.set_axis_off()
    plt.show()


def plot_band_difference_histogram(reference, modified, band):
    reference_array = reference[band].read(1).astype(np.float32).flatten()
    modified_array = modified[band].read(1).astype(np.float32).flatten()

    difference_array = reference_array - modified_array
    # current_band = current_band[current_band < 65000]
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.hist(
        difference_array,
        bins=100,
    )
    ax.set_title(f"{band}")
    plt.show()


def plot_band_histogram(
    reference: L2A_Band_Stack, bands=None, plot_title="Histogram of bands"
):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    for band in reference.read_bands(bands):
        reference_array = band.astype(np.float32).flatten()
        ax.hist(
            reference_array,
            bins=100,
            alpha=0.5,
            label=band,
        )
    ax.set_title(plot_title)
    ax.set_yscale("log")
    ax.legend()

    plt.show()


def plot_scl_as_bar_chart(reference: L2A_Band_Stack, title=None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    scl_array = reference["SCL"].read(1).astype(np.float32).flatten()
    for scene_class in class_dict:
        ax.bar(
            scene_class,
            np.sum(scl_array == class_dict[scene_class]["value"]),
            color=mcolors.to_hex(np.array(class_dict[scene_class]["color"]) / 255),
        )
    ax.set_title(f"Scene Classification for {title}")
    plt.xticks(rotation=45, ha="right")
    plt.show()


def plot_difference_histogram(
    reference: L2A_Band_Stack,
    modified: L2A_Band_Stack,
    bands=None,
    title="Histogram of differences",
):
    """
    Plot the histogram of the pixelwise difference between the reference l2a product
    and the modified l2a product.
    Arguments:
        reference: L2A_Band_Stack object of reference product
        modified: L2A_Band_Stack object of modified product
        bands: list of strings of bands
        plot_title: string of plot title
    """
    if bands is None:
        bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12"]
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    for band in bands:
        reference_array = reference[band].read(1).astype(np.float32).flatten()
        modified_array = modified[band].read(1).astype(np.float32).flatten()

        difference_array = reference_array - modified_array
        ax.hist(
            difference_array,
            bins=100,
            alpha=0.5,
            label=band,
        )
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend()
    plt.show()


def multiplot_difference_histogram(
    l2aa: L2A_Analysis, locs, mods, bands, share_axes=False
):
    """
    Plot the histogram of the pixelwise difference between the reference l2a product
    and the modified l2a product for multiple locations and modifications.
    Arguments:
        l2aa: L2A_Analysis object
        locs: list of strings of locations
        mods: list of strings of modifications
        bands: list of strings of bands
        axis_limits: tuple of axis limits
    """

    figsize = (len(locs) * 5, len(mods) * 4)
    fig, ax = plt.subplots(
        nrows=len(mods),
        ncols=len(locs),
        figsize=figsize,
        sharex=share_axes,
        sharey=share_axes,
    )
    for loc in locs:
        reference = l2aa.reference_bands[loc]
        for mod in mods:
            modified = l2aa.modified_bands[loc][mod]
            for band in bands:
                reference_array = reference[band].read(1).astype(np.float32).flatten()
                modified_array = modified[band].read(1).astype(np.float32).flatten()

                difference_array = reference_array - modified_array

                ax[mods.index(mod), locs.index(loc)].hist(
                    difference_array,
                    bins=100,
                    alpha=0.5,
                    label=band,
                )
                # if axis_limits is not None:
                #     ax[mods.index(mod), locs.index(loc)].set_xlim(axis_limits)
                ax[mods.index(mod), locs.index(loc)].set_title(f"{loc} {mod}")
                ax[mods.index(mod), locs.index(loc)].set_yscale("log")
                ax[mods.index(mod), locs.index(loc)].legend()


def get_stats(reference, modified, bands):
    """
    Get the stats for a single location and modification
    Arguments:
        reference: dictionary of reference bands
        modified: dictionary of modified bands
        bands: list of strings of bands
    Returns:
        stats: dataframe containing the mean, std div, min, max for each band
    """
    cols = ["mean", "std", "max", "min"]
    stats = pd.DataFrame(columns=cols)

    for band in bands:
        current_band = reference[band].read(1).flatten().astype(np.float32) - modified[
            band
        ].read(1).flatten().astype(np.float32)
        stats.loc[band] = [
            np.mean(current_band),
            np.std(current_band),
            np.max(current_band),
            np.min(current_band),
        ]
    return stats


def get_stats_average(reference, modified, bands):
    """
    Get the average stats for a single location and modification
    Arguments:
        reference: dictionary of reference bands
        modified: dictionary of modified bands
        bands: list of strings of bands
    Returns:
        stats: series containing the average mean, std div, min, max over all bands
    """
    cols = ["mean", "std", "max", "min"]
    stats = pd.DataFrame(columns=cols)
    for band in bands:
        current_band = reference[band].read(1).flatten().astype(np.float32) - modified[
            band
        ].read(1).flatten().astype(np.float32)
        stats.loc[band] = [
            np.mean(current_band),
            np.std(current_band),
            np.max(current_band),
            np.min(current_band),
        ]

    return stats.mean(axis=0)


def get_stats_average_multi(l2aa: L2A_Analysis, locs, mods, bands):
    """
    Get the average stats for a list of locations and modifications
    Arguments:
        l2aa: L2A_Analysis object
        locs: list of strings of locations
        mods: list of strings of modifications
        bands: list of strings of bands
    Returns:
        stats: dictionary of dataframes containing the average mean, std div, min, max for each location and modification
    """
    stats = {}
    dataframes = ["mean", "std", "max", "min"]
    for df in dataframes:
        stats[df] = pd.DataFrame(columns=mods)

    for loc in locs:
        reference = l2aa.reference_bands[loc]
        loc_means = pd.DataFrame(columns=mods)
        loc_stds = pd.DataFrame(columns=mods)
        loc_maxs = pd.DataFrame(columns=mods)
        loc_mins = pd.DataFrame(columns=mods)
        for band in bands:
            mod_means = []
            mod_stds = []
            mod_maxs = []
            mod_mins = []
            for mod in mods:
                modified = l2aa.modified_bands[loc][mod]
                current_band = reference[band].read(1).flatten().astype(
                    np.float32
                ) - modified[band].read(1).flatten().astype(np.float32)
                mod_means.append(np.mean(current_band))
                mod_stds.append(np.std(current_band))
                mod_maxs.append(np.max(current_band))
                mod_mins.append(np.min(current_band))
            loc_means.loc[band] = mod_means
            loc_stds.loc[band] = mod_stds
            loc_maxs.loc[band] = mod_maxs
            loc_mins.loc[band] = mod_mins
        stats["mean"].loc[loc] = loc_means.mean(axis=0)
        stats["std"].loc[loc] = loc_stds.mean(axis=0)
        stats["max"].loc[loc] = loc_maxs.mean(axis=0)
        stats["min"].loc[loc] = loc_mins.mean(axis=0)

    return stats


def plot_scl_in_rgb(rio_scl, title=None):
    # create a new array with the same shape as the original data
    # but with 3 channels (RGB)
    scl_array = rio_scl.read(1)
    rgb = np.zeros((rio_scl.height, rio_scl.width, 3), dtype=np.uint8)

    def rgb_to_hex(rgb: tuple):
        return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

    handles = []
    labels = []

    import matplotlib.patches as mpatches

    for scene_class in class_dict:
        rgb[scl_array == class_dict[scene_class]["value"]] = class_dict[scene_class][
            "color"
        ]
        handles.append(
            mpatches.Patch(
                color=rgb_to_hex(class_dict[scene_class]["color"]),
                label=class_dict[scene_class]["description"],
            )
        )
        labels.append(class_dict[scene_class]["description"])

    # plot the RGB image with a legend
    ax, fig = plt.subplots(1, 1, figsize=(10, 10))
    fig.imshow(rgb, interpolation="none")
    if title is not None:
        fig.set_title(f"Scene Classification for {title}")
    fig.set_axis_off()
    fig.legend(handles, labels, loc="lower right")
    plt.show()


def get_max_difference_to_l2a(l2aa: L2A_Analysis, loc, mods, bands):
    """
    Get the maximum difference between the reference l2a product and the modified l2a product
    Arguments:
        l2aa: L2A_Analysis object
        loc: string of location
        mods: list of strings of modifications
        bands: list of strings of bands
    Returns:
        max_diffs: dictionary of arrays containing the maximum difference for each band
        mod_ids: dictiondary of arrays containing the id of the modification with the maximum difference for each band
    """
    max_diffs = {}
    mod_ids = {}
    for band in bands:
        max_diffs[band] = np.zeros_like(l2aa.reference_bands[loc][band].read())
        mod_ids[band] = np.zeros_like(l2aa.reference_bands[loc][band].read())

    mod_id = 0
    for mod in mods:
        for band in bands:
            reference_array = l2aa.reference_bands[loc][band].read(1).astype(np.float32)
            modified_array = (
                l2aa.modified_bands[loc][mod][band].read(1).astype(np.float32)
            )

            difference_array = np.abs(reference_array - modified_array)
            max_diffs[band] = np.maximum(max_diffs[band], difference_array)
            mod_ids[band][difference_array == max_diffs[band]] = mod_id
        mod_id += 1
    return max_diffs, mod_ids


def export_to_jp2(product, bands, output, meta=None):
    """
    Convert bands of a product to jp2 format
    Arguments:
        product: iterable of nd arrays
        bands: list of strings of bands
        output: string of output path
    """
    if isinstance(bands, str):
        bands = [bands]
    if len(bands) > 3:
        raise ValueError("bands must be a list of length 1, 2 or 3")
    if meta is None:
        try:
            meta = product[bands[0]].meta
        except:
            raise ValueError("meta data must be provided")

    meta["count"] = len(bands)
    with rio.open(output, "w", **meta) as dst:
        for i, band in enumerate(bands):
            dst.write(product[band], i + 1)


def comparison_matrix(l2aa: L2A_Analysis, loc, mods, bands):
    """
    Get the comparison matrix of modifications for a single location
    Arguments:
        l2aa: L2A_Analysis object
        loc: string of location
        mods: list of strings of modifications
        bands: list of strings of bands
    Returns:
        comparison_matrix: dataframe containing the average difference between each modification
    """
    comparison_matrix = pd.DataFrame(columns=mods, index=mods)
    for mod1 in mods:
        for mod2 in mods:
            if mod1 == mod2:
                comparison_matrix.loc[mod1, mod2] = 0
            else:
                comparison_matrix.loc[mod1, mod2] = get_stats_average(
                    l2aa.modified_bands[loc][mod1],
                    l2aa.modified_bands[loc][mod2],
                    bands,
                )["mean"]
    return comparison_matrix.astype(np.float32)


def get_comparison_matrices(l2aa: L2A_Analysis, loc, mods, bands):
    """
    Get the comparison matricies of modifications for a single location
    Args:
        l2aa: L2A_Analysis object
        loc: string of location
        mods: list of strings of modifications
        bands: list of strings of bands
    Returns:
        comparison_matrix: dataframe containing the mean, std div, min & max difference between each modification
    """
    comp_mean = pd.DataFrame(columns=mods, index=mods)
    comp_std = pd.DataFrame(columns=mods, index=mods)
    comp_max = pd.DataFrame(columns=mods, index=mods)
    comp_min = pd.DataFrame(columns=mods, index=mods)

    for mod1 in mods:
        for mod2 in mods:
            if mod1 == mod2:
                comp_mean.loc[mod1, mod2] = 0
                comp_std.loc[mod1, mod2] = 0
                comp_max.loc[mod1, mod2] = 0
                comp_min.loc[mod1, mod2] = 0
            else:
                stats = get_stats(
                    l2aa.modified_bands[loc][mod1],
                    l2aa.modified_bands[loc][mod2],
                    bands,
                )
                comp_mean.loc[mod1, mod2] = stats["mean"].mean()
                comp_std.loc[mod1, mod2] = stats["std"].mean()
                comp_max.loc[mod1, mod2] = stats["max"].mean()
                comp_min.loc[mod1, mod2] = stats["min"].mean()

    # combine the dataframes into a single dataframe
    comparsion_matrix = pd.concat(
        [comp_mean, comp_std, comp_max, comp_min],
        keys=["mean", "std", "max", "min"],
        axis=1,
    )
    return comparsion_matrix.astype(np.float32)


def plot_comparison_matrix(comp_matrix):
    """
    Plot the comparison matrix with the modification names as labels
    Arguments:
        comp_matrix: dataframe containing the average difference between each modification
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    im = ax.imshow(comp_matrix, cmap="cividis")
    ax.set_xticks(np.arange(len(comp_matrix.columns)))
    ax.set_yticks(np.arange(len(comp_matrix.columns)))
    ax.set_xticklabels(comp_matrix.columns)
    ax.set_yticklabels(comp_matrix.columns)
    ax.set_title("Comparison Matrix")
    # add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("difference", rotation=-90, va="bottom")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.show


def get_modification_names(modification_dicts):
    """
    Get the names of the modifications from the modification dictionarys
    Arguments:
        modification_dict: list of dictionaries of modifications
            Format:
            [
                {
                    "flag" : "enviroment variable",
                    "value" : "value of enviroment variable",
                    "description" : "description of modification",
                    "name" : "name of modification"
                },
                ...
            ]
    Returns:
        mod_names: list of strings of modification names
    """
    mod_names = []
    for mod in modification_dicts:
        mod_names.append(mod["name"])
    return mod_names
