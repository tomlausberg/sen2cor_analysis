import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from l2a_analysis import L2A_Band, L2A_Analysis


def plot_true_color_image(product, gamma=1.5, title="True Color Image", clip_value=65536):
    blue = product["B02"].read() 
    green = product["B03"].read() 
    red = product["B04"].read() 
    rgb = np.dstack((red, green, blue))

    # clip values
    rgb[rgb > clip_value] = clip_value
    # normalize rgb
    rgb = rgb / rgb.max()

    #do gamma transformation
    rgb_gamma = np.power(rgb, 1/gamma)


    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(
        rgb,
        interpolation="none",
    )
    ax.set_title(title)
    ax.set_axis_off()
    plt.show()


def plot_rgb_image(product, red_band, green_band, blue_band):
    brightness = 5.0
    blue = product[blue_band].read(1) * brightness / 65536
    green = product[green_band].read(1) * brightness / 65536
    red = product[red_band].read(1) * brightness / 65536

    rgb = np.dstack((red, green, blue))
    print(f"Max value: {np.max(rgb)}")
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(
        rgb,
        interpolation="none",
    )
    ax.set_title(f"False colour RGB with bands {red_band}, {green_band}, {blue_band}")
    ax.set_axis_off()
    plt.show()


def plot_band(product, band, color_map="Blues"):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
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


def plot_band_histogram(reference, bands, plot_title="Histogram of bands"):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    for band in bands:
        reference_array = reference[band].read(1).astype(np.float32).flatten()

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


def plot_difference_histogram(
    reference, modified, bands, plot_title="Histogram of differences"
):
    """
    Plot the histogram of the pixelwise difference between the reference l2a product
    and the modified l2a product.
    Arguments:
        reference: dictionary of reference bands
        modified: dictionary of modified bands
        bands: list of strings of bands
        plot_title: string of plot title
    """
    bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12"]
    # bands = ['SCL', 'AOT', 'TCI']
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
    ax.set_title(plot_title)
    ax.set_yscale("log")
    ax.legend()
    plt.show()


def multiplot_difference_histogram(l2aa: L2A_Analysis, locs, mods, bands):
    """
    Plot the histogram of the pixelwise difference between the reference l2a product
    and the modified l2a product for multiple locations and modifications.
    Arguments:
        l2aa: L2A_Analysis object
        locs: list of strings of locations
        mods: list of strings of modifications
        bands: list of strings of bands
    """
    fig, ax = plt.subplots(nrows=len(mods), ncols=len(locs), figsize=(15, 15))
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


def plot_scl_in_rgb(rio_scl):
    # map the data to the following colors
    colour_dict = {
        0: (0, 0, 0),  # black: no data
        1: (255, 0, 0),  # red: saturated or defective
        2: (50, 50, 50),  # dark grey: casted shadows
        3: (165, 42, 42),  # brown: cloud shadows
        4: (0, 100, 0),  # green: vegetation
        5: (255, 255, 0),  # yellow: bare soils
        6: (0, 0, 255),  # blue: water
        7: (128, 128, 128),  # grey: unclassified
        8: (211, 211, 211),  # light grey: cloud medium probability
        9: (255, 255, 255),  # white: cloud high probability
        10: (0, 255, 255),  # light blue: thin cirrus
        11: (255, 192, 203),  # pink: snow
    }

    # create a new array with the same shape as the original data
    # but with 3 channels (RGB)
    scl_array = rio_scl.read(1)
    rgb = np.zeros((rio_scl.height, rio_scl.width, 3), dtype=np.uint8)

    # map the data to RGB
    for k, v in colour_dict.items():
        rgb[scl_array == k] = v

    # plot the RGB image
    import matplotlib.pyplot as plt

    plt.imshow(rgb)
    plt.show()


def get_max_difference_to_l2a(l2aa: L2A_Analysis, loc, mods, bands):
    pass