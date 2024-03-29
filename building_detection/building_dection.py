import os
import shutil
import json
import subprocess
from typing import Any
import zipfile
from datetime import datetime
import math
from pathlib import Path

import rasterio as rio
import rasterio.features
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import numpy as np
import pyproj
from osgeo import gdal
import xml.etree.ElementTree as ET

from copernicus import Client


def reproject_geojson(geojson_fn, src_crs, tgt_crs, out_fn=None):
    # Read the GeoJSON file
    with open(geojson_fn) as f:
        data = json.load(f)

    # Define the projection transformer
    transformer = pyproj.Transformer.from_crs(src_crs, tgt_crs, always_xy=True)

    # Convert the coordinates of each feature in the GeoJSON
    for feature in data["features"]:
        geometry = feature["geometry"]
        if geometry["type"] == "Point":
            x, y = geometry["coordinates"]
            x_new, y_new = transformer.transform(x, y)
            geometry["coordinates"] = [x_new, y_new]
        elif geometry["type"] == "LineString" or geometry["type"] == "MultiPoint":
            new_coords = []
            for x, y in geometry["coordinates"]:
                x_new, y_new = transformer.transform(x, y)
                new_coords.append([x_new, y_new])
            geometry["coordinates"] = new_coords
        elif geometry["type"] == "Polygon" or geometry["type"] == "MultiLineString":
            new_coords = []
            for ring in geometry["coordinates"]:
                new_ring = []
                for x, y in ring:
                    x_new, y_new = transformer.transform(x, y)
                    new_ring.append([x_new, y_new])
                new_coords.append(new_ring)
            geometry["coordinates"] = new_coords
        elif geometry["type"] == "MultiPolygon":
            new_coords = []
            for poly in geometry["coordinates"]:
                new_poly = []
                for ring in poly:
                    new_ring = []
                    for x, y in ring:
                        x_new, y_new = transformer.transform(x, y)
                        new_ring.append([x_new, y_new])
                    new_poly.append(new_ring)
                new_coords.append(new_poly)
            geometry["coordinates"] = new_coords

    # Change the CRS of the GeoJSON
    data["crs"] = {"type": "name", "properties": {"name": tgt_crs}}

    # Write the GeoJSON file
    if out_fn is None:
        out_fn = geojson_fn
    with open(out_fn, "w") as f:
        json.dump(data, f)


def rasterize_labels(geojson_fn, raster_fn, extent):
    resolution = 10980
    data_source = gdal.OpenEx(geojson_fn, gdal.OF_VECTOR)
    burn_value = 1
    no_data_value = 0
    data_type = "UInt16"
    output_format = "GTiff"

    cmd = f"gdal_rasterize  -burn {burn_value} -ts {resolution} {resolution} -a_nodata {no_data_value} -te {extent[0]} {extent[1]} {extent[2]} {extent[3]} -ot {data_type} -of {output_format} {geojson_fn} {raster_fn}"
    subprocess.call(cmd, shell=True)


def crop_to_extent(geojson_fn, raster_fn, out_fn):
    # get extent of geojson
    vector_ds = gdal.OpenEx(geojson_fn, gdal.OF_VECTOR)
    vector_extent = vector_ds.GetLayer().GetExtent()
    vector_crs = pyproj.CRS.from_wkt(vector_ds.GetLayer().GetSpatialRef().__str__())
    ds = gdal.Open(raster_fn)
    raster_crs = pyproj.CRS.from_wkt(ds.GetProjection())
    # convert extent to raster crs
    transformer = pyproj.Transformer.from_crs(vector_crs, raster_crs, always_xy=True)
    raster_extent1 = transformer.transform(vector_extent[0], vector_extent[3])
    raster_extent2 = transformer.transform(vector_extent[1], vector_extent[2])
    raster_extent = (
        raster_extent1[0],
        raster_extent1[1],
        raster_extent2[0],
        raster_extent2[1],
    )
    # crop raster to extent of geojson
    gdal.Translate(out_fn, ds, projWin=raster_extent)


def crop_to_extent2(extent_fn, raster_fn, out_fn):
    # get extent
    with rio.open(extent_fn) as ds:
        extent = ds.bounds
        extent_crs = ds.crs
    with rio.open(raster_fn) as ds:
        # transform extent to raster crs
        transformer = pyproj.Transformer.from_crs(extent_crs, ds.crs, always_xy=True)
        raster_extent1 = transformer.transform(extent[0], extent[3])
        raster_extent2 = transformer.transform(extent[1], extent[2])
        raster_extent = (
            raster_extent1[0],
            raster_extent1[1],
            raster_extent2[0],
            raster_extent2[1],
        )
    # crop raster to extent
    with rio.open(out_fn) as out_ds:
        out_ds.write(
            ds.read(
                window=rio.windows.from_bounds(*raster_extent, ds.transform), indexes=1
            )
        )
    # gdal.Translate(out_fn, raster_fn, projWin=raster_extent)


def get_credentials():
    with open("credentials.json") as f:
        data = json.load(f)
    return data["username"], data["password"]


def get_labels(train_path):
    labels = []
    for train_dir in os.listdir(train_path):
        geojson_filename = (
            f"global_monthly_2020_01_mosaic_{train_dir}_Buildings.geojson"
        )
        label_geojson = f"{train_path}/{train_dir}/labels/{geojson_filename}"
        if os.path.isfile(label_geojson):
            labels.append(label_geojson)
        # else:
        #     print(f"Label file {label_geojson} not found")
    return labels

def convert_band_images_to_ndarray(image_dir, bands):
    with rio.open(image_dir + "B02.tif") as src:
        image = src.read(1)
    image_dims = image.shape
    x_image = np.zeros((image_dims[0] * image_dims[1], len(bands)), dtype=np.float32)
    for i, band in enumerate(bands):
        with rio.open(image_dir + band + ".tif") as src:
            image = src.read(1)
        x_image[:, i] = image.flatten()
    return x_image


def convert_labels_to_raster_file(labels: np.ndarray, orginal_label_file, raster_file):
    """Converts 1d array to a 2d raster using rasterio"""
    with rio.open(orginal_label_file, "r", driver="GTiff") as src:
        shape = src.shape
        meta = src.meta
    labels = labels.reshape(shape)
    with rio.open(raster_file, "w", **meta) as dst:
        dst.write(labels, 1)


def convert_geojson_to_utm_raster(vector_file, raster_file, resolution=10):
    gdf = gpd.read_file(vector_file)

    # convert to utm
    gdf = gdf.to_crs(gdf.estimate_utm_crs())

    # Define the bounding box and resolution for the raster image
    bounds = gdf.total_bounds

    # Create a raster dataset
    with rio.Env():
        with rio.open(
            raster_file,
            "w",
            driver="GTiff",
            height=int((bounds[3] - bounds[1]) / resolution),
            width=int((bounds[2] - bounds[0]) / resolution),
            count=1,
            dtype="uint8",
            crs=gdf.crs,
            transform=rio.Affine(
                resolution, 0, int(bounds[0]), 0, -resolution, int(bounds[3])
            ),
        ) as dst:
            # Rasterize the GeoDataFrame to the created raster dataset
            mask = rio.features.geometry_mask(
                gdf.geometry, out_shape=dst.shape, transform=dst.transform, invert=True
            )
            dst.write(mask.astype("uint8"), 1)


def get_GRANULE_dir(SAFE_dir):
    """Returns the path to the GRANULE directory in the SAFE directory"""
    return SAFE_dir / "GRANULE" / os.listdir(SAFE_dir / "GRANULE")[0]


def get_SAFE_extent(SAFE_dir):
    """Returns the extent of the SAFE directory in the form [ulx, lry, lrx, uly]"""
    MTD_TL_xml_path = str(get_GRANULE_dir(SAFE_dir) / "MTD_TL.xml")
    root = ET.parse(MTD_TL_xml_path).getroot()
    tile_geocoding_el = root.findall(".//Tile_Geocoding")[0]
    size_el = tile_geocoding_el.findall(".//Size[@resolution='10']")[0]
    nrow = int(size_el.find("NROWS").text)
    ncol = int(size_el.find("NCOLS").text)
    geoposition_element = root.findall(".//Geoposition[@resolution='10']")[0]
    ulx = int(geoposition_element.find("ULX").text)
    uly = int(geoposition_element.find("ULY").text)
    lrx = ulx + ncol * 10
    lry = uly - nrow * 10
    return [ulx, lry, lrx, uly]


def get_region_of_interest(raster_label, raster_SAFE_dir):
    """Returns the region of interest for a safe file for a corresponding label raster file

    Parameters:
    raster_label (str): path to label raster file
    raster_SAFE_dir (str): path to safe directory

    Returns:
    list: [row0, col0, nrow_win, ncol_win] to be used in GIPP
    """

    label_ds = gdal.Open(str(raster_label))
    label_extent = label_ds.GetGeoTransform()
    label_ulx = label_extent[0]
    label_uly = label_extent[3]
    label_lrx = label_ulx + label_ds.RasterXSize * label_extent[1]
    label_lry = label_uly + label_ds.RasterYSize * label_extent[5]

    del label_ds

    SAFE_ulx, SAFE_lry, SAFE_lrx, SAFE_uly = get_SAFE_extent(raster_SAFE_dir)

    row0 = math.floor((SAFE_uly - label_uly) / 10)
    col0 = math.floor((label_ulx - SAFE_ulx) / 10)
    nrow_win = math.ceil((label_uly - label_lry) / 10)
    ncol_win = math.ceil((label_lrx - label_ulx) / 10)

    print(
        f"label_extent: ulx: {label_ulx}\tuly: {label_uly}\tlrx: {label_lrx}\tlry: {label_lry}"
    )
    print(
        f"SAFE_extent: ulx:  {SAFE_ulx}\tuly: {SAFE_uly}\tlrx: {SAFE_lrx}\tlry: {SAFE_lry}"
    )
    return [row0, col0, nrow_win, ncol_win]


def transform_roi_for_sen2cor(roi):
    """

    Parameters:
        roi (list): [first_row, first_col, nrow, ncol]

    Returns:
        sen2cor_roi (list): [row0, col0, nrow_win, ncol_win]
            row0, col0: specifies the midpoint of the region of interest
            nrow_win, ncol_win defines a rectangle around the midpoint in pixel
            row0, col0, nrow_win and ncol_win must be integer divisible by 6, to prevent rounding errors for downsampling
            specify always a 10m resolution ROI, it will be automatically adapted to the lower resolutions

    """
    buffer = 24  # 12 pixel buffer around the roi
    div = 12
    nrow_win = (
        math.ceil(roi[2] / div) * div + buffer
    )  # make sure nrow_win is divisible by 6
    ncol_win = math.ceil(roi[3] / div) * div + buffer

    row0 = roi[0] + math.floor(roi[2] / 2)
    row0 = row0 - (row0 % div)  # make sure row0 is divisible by 6
    col0 = roi[1] + math.floor(roi[3] / 2)
    col0 = col0 - (col0 % div)  # make sure col0 is divisible by 6

    if row0 < nrow_win or col0 < ncol_win:
        raise Exception("ROI not covered by image")

    sen2cor_roi = [str(row0), str(col0), str(nrow_win), str(ncol_win)]

    return sen2cor_roi


class SN7_Location(object):
    """Class to create and manage a directory for a SN7 location
    Example directory structure:
    SN7_Location (e.g. "1025E")
    ├── L1C.SAFE
    ├── IMG_DATA
    │   ├── Reference
    │   │   ├── R10m, R20m, R60m
    │   ├── Mod1, ...
    │   │   ├── R10m, R20m, R60m
    ├── metadata.json
    ├── raster_labels.tif
    """

    def __init__(self, path: Path, label) -> None:
        self.path = path
        self.metadata_json = self.path / "metadata.json"
        self.l1c_path = Path("None")
        self.l2a_path = Path("None")
        self.raster_labels_path = Path("None")
        self.vector_labels_path = Path("None")

        self.r10m_bands = ["B02", "B03", "B04", "B08", "AOT", "WVP"]
        self.r20m_bands = ["B05", "B06", "B07", "B8A", "B11", "B12", "SCL"]

        self.l2a_mods = []
        self.roi = None

        self.label = label

        self.product_name = None
        self.product_id = None

        if not os.path.exists(path):
            os.makedirs(path)

        if os.path.isfile(self.metadata_json):
            self.load_metadata()
        else:
            self.update_metadata()

    def __str__(self) -> str:
        return f"Location({self.label})"

    def __repr__(self) -> str:
        return f"Location({self.label})"

    def init_labels(self, vector_label_path):
        """
        Setup raster labels for the location
        - reproject to UTM
        - rasterize
        - copy to location
        """
        if self.vector_labels_path != Path("None"):
            print(f"Labels already set")
        else:
            self.vector_labels_path = vector_label_path

        if self.raster_labels_path != Path("None"):
            print(f"Raster labels already set")
        else:
            self.raster_labels_path = self.path / "raster_labels.tif"
            convert_geojson_to_utm_raster(vector_label_path, self.raster_labels_path)

        self.update_metadata()

    def init_l1c(self, client: Client, download_path):
        start_date = "2020-01-01"
        end_date = "2020-01-31"
        platform = "SENTINEL-2"

        # Get product id
        if self.vector_labels_path == Path("None"):
            raise Exception("Labels not set")
        if self.product_id is None:
            ds = gdal.OpenEx(str(self.vector_labels_path), gdal.OF_VECTOR)
            extent = ds.GetLayer().GetExtent()
            footprint = f"POLYGON(({extent[0]} {extent[2]}, {extent[1]} {extent[2]}, {extent[1]} {extent[3]}, {extent[0]} {extent[3]}, {extent[0]} {extent[2]}))"
            prod_list = client.search(start_date, end_date, platform, footprint)
            for index, row in prod_list.iterrows():
                if "MSIL1C" in row["Name"]:  # Only use L1C products
                    self.product_id = row["Id"]
                    self.product_name = row["Name"]
                    break
        else:
            print(f"Product {self.product_id} already found")
        if self.product_id is None:
            raise Exception("No product found")

        # Download product
        if self.l1c_path == Path("None"):
            SAFE_download_path = download_path / f"{self.label}.zip"
            if SAFE_download_path.exists() == False:
                print(f"Downloading {self.product_name}, {self.product_id}...")
                client.download(self.product_id, SAFE_download_path)
            else:
                print(f"Product {self.product_name} already downloaded")

            self.set_l1c(SAFE_download_path)
        else:
            print(f"Product {self.product_name} already downloaded")

        self.roi = self.get_region_of_interest()
        self.update_metadata()

    def set_l1c(self, l1c_zip_path):
        """
        Setup L1C for the location
        - unzip
        - move to location
        """
        self.l1c_path = self.path / self.product_name

        extract_dir = self.path / "extract"
        extract_dir.mkdir(exist_ok=True)

        with zipfile.ZipFile(l1c_zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        # rename .SAFE file to L1C.SAFE
        os.rename(extract_dir / self.product_name, self.l1c_path)
        os.rmdir(extract_dir)

        self.update_metadata()

    def set_l2a_data(self, l2a_path):
        """
        Extract L2A for the location
        - find mods in l2a_dir
        - copy bands to sample directory
        """

        os.makedirs(self.path / "IMG_DATA", exist_ok=True)
        self.l2a_path = l2a_path
        self.mods = [
            mod for mod in os.listdir(l2a_path) if os.path.isdir(l2a_path / mod)
        ]

        for mod in self.mods:
            mod_path = self.path / "IMG_DATA" / mod
            os.makedirs(mod_path, exist_ok=True)
            img_data_path = (
                l2a_path / mod / get_GRANULE_dir(l2a_path / mod) / "IMG_DATA"
            )

            # copy to /IMG_DATA/mod
            for res in os.listdir(img_data_path):
                shutil.copytree(img_data_path / res, mod_path / res)

        self.update_metadata()

    def update_metadata(self):
        with open(self.metadata_json, "w") as f:
            metadata = {
                "l1c_path": str(self.l1c_path),
                "l2a_path": str(self.l2a_path),
                "vector_labels_path": str(self.vector_labels_path),
                "raster_labels_path": str(self.raster_labels_path),
                "path": str(self.path),
                "l2a_mods": self.l2a_mods,
                "roi": self.roi,
                "label": self.label,
                "product_name": self.product_name,
                "product_id": self.product_id,
            }
            json.dump(metadata, f, indent=4)

    def load_metadata(self):
        with open(self.metadata_json, "r") as f:
            metadata = json.load(f)
            self.l1c_path = Path(metadata["l1c_path"])
            self.l2a_path = Path(metadata["l2a_path"])
            self.vector_labels_path = Path(metadata["vector_labels_path"])
            self.raster_labels_path = Path(metadata["raster_labels_path"])
            self.l2a_mods = metadata["l2a_mods"]
            self.roi = metadata["roi"]
            self.label = metadata["label"]
            self.product_name = metadata["product_name"]
            self.product_id = metadata["product_id"]

    def get_region_of_interest(self):
        roi = get_region_of_interest(self.raster_labels_path, self.l1c_path)
        return transform_roi_for_sen2cor(roi)

    def init_l2a(self, l2aa):
        self.l2a_path = Path(l2aa.report_dir) / self.path.name
        self.l2a_mods = os.listdir(self.l2a_path)
        for mod in self.l2a_mods:
            mod_safe_path = self.l2a_path / mod / os.listdir(self.l2a_path / mod)[0]

            r10m_src = get_GRANULE_dir(mod_safe_path) / "IMG_DATA" / "R10m"
            r10m_dst = self.path / "IMG_DATA" / mod / "R10m"
            r10m_dst.mkdir(parents=True, exist_ok=True)

            r20m_src = get_GRANULE_dir(mod_safe_path) / "IMG_DATA" / "R20m"
            r20m_dst = self.path / "IMG_DATA" / mod / "R20m"
            r20m_dst.mkdir(parents=True, exist_ok=True)

            print(f"Copy bands from {r10m_src} to {r10m_dst}")
            for f in os.listdir(r10m_src):
                print(f"\tCopy {r10m_src / f} \n\tto {r10m_dst}")
            print(f"Copy bands from {r20m_src} to {r20m_dst}")

            for band in self.r10m_bands:
                found = False
                for image in os.listdir(r10m_src):
                    file_name = f"{band}_10m.jp2"
                    if file_name in image:
                        image_path = r10m_src / image
                        shutil.copy(image_path, r10m_dst)
                        print(f"Copy {image_path} to {r10m_dst}")
                        found = True
                        break
                if not found:
                    print(f"WARNING: Could not find {band}_10m.jp2 in {r10m_src}")

            for band in self.r20m_bands:
                for image in os.listdir(r20m_src):
                    if f"{band}_20m.jp2" in image:
                        image_path = r20m_src / image
                        shutil.copy(image_path, r20m_dst)
                        print(f"Copy {image_path} to {r20m_dst}")
                        break
                print(f"WARNING: Could not find {band}_20m.jp2 in {r20m_src}")

    def get_sample_points(self, number_of_points, ratio):
        """
        Parameters:
            number_of_points (int): number of points to sample
            ratio (float): ratio of positive to negative samples

        Returns:
            samples: (geopandas.GeoDataFrame): gdf of sample points in the form [row, col, label]
        """

        with rio.open(self.raster_labels_path) as ds:
            label = ds.read(1)
            label_shape = label.shape
            # label = label.flatten()

        # get positive and negative samples
        positive_samples = np.argwhere(label == 1)
        negative_samples = np.argwhere(label == 0)

        # sample positive and negative samples
        positive_samples = positive_samples[
            np.random.choice(positive_samples.shape[0], int(number_of_points * ratio))
        ]
        negative_samples = negative_samples[
            np.random.choice(
                negative_samples.shape[0], int(number_of_points * (1 - ratio))
            )
        ]

        # add label to samples
        positive_samples = np.concatenate(
            (positive_samples, np.ones((positive_samples.shape[0], 1))), axis=1
        )

        negative_samples = np.concatenate(
            (negative_samples, np.zeros((negative_samples.shape[0], 1))), axis=1
        )

        # combine positive and negative samples
        samples = np.concatenate((positive_samples, negative_samples), axis=0)

        # add Point geometry to samples
        geometries = np.array(
            [
                Point(
                    rio.transform.xy(
                        ds.transform, sample[0], sample[1], offset="center"
                    )
                )
                for sample in samples
            ]
        )

        # convert to gdf
        samples_gdf = gpd.GeoDataFrame(samples, columns=["row", "col", "label"])
        samples_gdf.set_geometry(geometries, inplace=True)

        # set crs
        samples_gdf.crs = ds.crs
        return samples_gdf

    def get_samples(self, number_of_points, ratio, mod):
        """
        Parameters:
            number_of_points (int): number of points to sample
            ratio (float): ratio of positive to negative samples
            mod (str): mod to sample from

        Returns:
            samples: (geopandas.GeoDataFrame): gdf of sample points in the form [row, col, label, band1, band2, ...]
        """

        samples = self.get_sample_points(number_of_points, ratio)
        coord_list = [(x, y) for x, y in zip(samples["geometry"].x, samples["geometry"].y)]
        # add bands to samples

        for band in self.r10m_bands:
            # get the path of the file name containing the band
            band_path = self.path / "IMG_DATA" / mod / "R10m"
            band_path = band_path / [f for f in os.listdir(band_path) if band in f][0]

            with rio.open(band_path) as ds:
                # get the value of the band at the sample points
                samples[band] = [x[0] for x in ds.sample(coord_list)]

        for band in self.r20m_bands:
            # get the path of the file name containing the band
            try:
                band_path = self.path / "IMG_DATA" / mod / "R20m"
                band_path = band_path / [f for f in os.listdir(band_path) if band in f][0]
                with rio.open(band_path) as ds:
                    # get the value of the band at the sample points
                    samples[band] = [x[0] for x in ds.sample(coord_list)]
            except IndexError:
                print(f"Could not find {band} in {self.path / 'IMG_DATA' / mod / 'R20m'}")
                continue
                # raise FileNotFoundError(f"Could not find {band} in {self.path / 'IMG_DATA' / mod / 'R20m'}")


        return samples
    
def convert_gdf_to_ndarray(gdf):
    """
    Parameters:
        gdf (geopandas.GeoDataFrame): gdf of sample points in the form [row, col, label, band1, band2, ...]
    Returns:
        samples: (np.ndarray): array of sample points in the form [label, band1, band2, ...]
    """

    samples = gdf.drop(columns=["geometry", "row", "col"])
    return samples.to_numpy()


def create_dataset( sn7_locations, number_of_points, ratio, mod):
    """
    Parameters:
        sn7_locations (list): list of SN7_Location objects
        number_of_points (int): number of points to sample
        ratio (float): ratio of positive to negative samples
        mod (str): mod to sample from

    Returns:
        dataset: (np.ndarray): array of sample points in the form [label, band1, band2, ...]
    """
    dataset = []
    for location in sn7_locations:
        location_samples = location.get_samples(number_of_points / len(sn7_locations), ratio, mod)
        sample_array = convert_gdf_to_ndarray(location_samples)
        dataset.append(sample_array)
    
    dataset = np.concatenate(dataset, axis=0)
    return dataset
