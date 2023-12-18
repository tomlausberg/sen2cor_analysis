import os
import json
import subprocess
import zipfile
from datetime import datetime
import math

import rasterio as rio
import rasterio.features
import geopandas as gpd
import numpy as np
import pyproj
from osgeo import gdal
import xml.etree.ElementTree as ET

from sentinelsat import (
    SentinelAPI,
    read_geojson,
    geojson_to_wkt,
    LTAError,
    LTATriggered,
)


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


def get_sentinel2_product_id(footprint, date=("20231101", "20231120")):
    """Takes a WKT footprint and returns the product id of the least cloudy Sentinel-2 product or None if no product is found."""
    user, password = get_credentials()
    api = SentinelAPI(user, password, "https://scihub.copernicus.eu/dhus/")

    products = api.query(
        footprint,
        date=date,
        platformname="Sentinel-2",
        processinglevel="Level-1C",
        cloudcoverpercentage=(0, 30),
        area_relation="Contains",
    )

    products_df = api.to_dataframe(products)
    if products_df.shape[0] > 0:
        products_df_sorted = products_df.sort_values(
            "cloudcoverpercentage", ascending=True
        )
        print(f"Found product {products_df_sorted.index[0]}")
        return products_df_sorted.index[0]
    else:
        print("No product found")
        return None


def get_sentinel2_product_ids(labels):
    """Takes a list of geojson labels and returns a dictionary of product ids and labels."""
    products = {}
    for label in labels:
        try:
            label_name = label.split("_mosaic_")[-1].split("_Buildings")[0]
            print(label_name, end=": ")
            ds = gdal.OpenEx(label, gdal.OF_VECTOR)
            extent = ds.GetLayer().GetExtent()
            footprint = f"POLYGON(({extent[0]} {extent[2]}, {extent[1]} {extent[2]}, {extent[1]} {extent[3]}, {extent[0]} {extent[3]}, {extent[0]} {extent[2]}))"
            print(footprint)
            products[label_name] = get_sentinel2_product_id(footprint)
        except Exception as e:
            print(f"Error: {e}")
            continue
            # raise e
    return products


# def download_sentinel2_products(id_dict, download_path):
#     downloaded_products = []
#     user, password = get_credentials()
#     api = SentinelAPI(user, password, "https://apihub.copernicus.eu/apihub/")

#     for label_name, product_id in id_dict.items():
#         data_dir = os.path.join(download_path, label_name)
#         if os.path.isfile(os.path.join(data_dir, "product_info.json")) == False:
#             try:
#                 os.mkdir(data_dir)
#             except FileExistsError:
#                 pass
#             try:
#                 product_info = api.download(
#                     [product_id], directory_path=download_path, checksum=True
#                 )
#             except LTATriggered as e:
#                 print(f"LTATriggered: {e}")
#                 continue
#             # extact zip
#             zip_path = os.path.join(download_path, product_info["title"] + ".zip")
#             with zipfile.ZipFile(zip_path, "r") as zip_ref:
#                 zip_ref.extractall(download_path)
#                 # remove zip
#             os.remove(zip_path)
#             # move .SAFE to labeled folder
#             safe_path = f"{data_dir}/{product_info['title']}.SAFE"
#             os.rename(f"{download_path}/{product_info['title']}.SAFE", safe_path)
#             # update product_info
#             product_info["label"] = label_name
#             product_info["path"] = safe_path
#             downloaded_products.append(product_info)
#             # save product_info as json
#             json_dump = product_info.copy()
#             # serialize datetimes
#             json_dump["date"] = json_dump["date"].strftime("%Y-%m-%d %H:%M:%S")
#             json_dump["Creation Date"] = json_dump["Creation Date"].strftime(
#                 "%Y-%m-%d %H:%M:%S"
#             )
#             json_dump["Ingestion Date"] = json_dump["Ingestion Date"].strftime(
#                 "%Y-%m-%d %H:%M:%S"
#             )
#             with open(os.path.join(data_dir, "product_info.json"), "w") as f:
#                 json.dump(json_dump, f)
#             print(f"Downloaded {product_id} to {download_path}")
#         else:
#             print(f"Product {product_id} already downloaded")
#             with open(os.path.join(data_dir, "product_info.json"), "r") as f:
#                 product_info = json.load(f)
#                 downloaded_products.append(product_info)

#     return downloaded_products


def get_sentinel2_products(id_dict, download_path):
    user, password = get_credentials()
    api = SentinelAPI(
        user, password, "https://apihub.copernicus.eu/apihub/", show_progressbars=True
    )

    product_ids = []

    for label_name, product_id in id_dict.items():
        if (
            product_id is not None
            and os.path.exists(os.path.join(download_path, product_id)) is False
        ):
            product_ids.append(product_id)
            # check if product is online
            try:
                product_info = api.get_product_odata(product_id)
            except LTAError as e:
                print(f"LTAError: {e}")
                continue
            print(
                f"Product {product_id}: {'Online' if product_info['Online'] else 'Offline'}"
            )
        else:
            print(f"Product {product_id} already downloaded")

    product_infos = api.download_all(
        product_ids,
        directory_path=download_path,
        checksum=True,
    )

    invert_products = {v: k for k, v in id_dict.items()}

    for product_info in product_infos.downloaded.values():
        print(product_info)
        zip_path = os.path.join(download_path, product_info["title"] + ".zip")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(download_path)
        # remove zip
        # os.remove(zip_path)
        # move .SAFE to labeled folder
        product_info["date"] = product_info["date"].strftime("%Y-%m-%d %H:%M:%S")
        product_info["Creation Date"] = product_info["Creation Date"].strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        product_info["Ingestion Date"] = product_info["Ingestion Date"].strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        product_info["label"] = invert_products[product_info["id"]]
        product_info["path"] = os.path.join(
            download_path, product_info["title"] + ".SAFE"
        )
    return product_infos.downloaded.values()


def convert_band_images_to_ndarray(image_dir, bands):
    with rio.open(image_dir + "B02.tif") as src:
        image = src.read(1)
    image_dims = image.shape
    x_image = np.zeros(( image_dims[0] * image_dims[1], len(bands)), dtype=np.float32)
    for i, band in enumerate(bands):
        with rio.open(image_dir + band + ".tif") as src:
            image = src.read(1)
        x_image[:,i] = image.flatten()
    return x_image

def convert_labels_to_raster_file( labels: np.ndarray, orginal_label_file, raster_file):
    """ Converts 1d array to a 2d raster using rasterio """
    with rio.open(orginal_label_file, 'r', driver='GTiff') as src:
        shape = src.shape
        meta = src.meta
    labels = labels.reshape(shape)
    with rio.open(raster_file, 'w', **meta) as dst:
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
                raster_file, 'w',
                driver='GTiff',
                height=int((bounds[3] - bounds[1]) / resolution),
                width=int((bounds[2] - bounds[0]) / resolution),
                count=1,
                dtype='uint8',
                crs=gdf.crs,
                transform=rio.Affine(resolution, 0, int(bounds[0]), 0, -resolution, int(bounds[3]))
        ) as dst:
            # Rasterize the GeoDataFrame to the created raster dataset
            mask = rio.features.geometry_mask(gdf.geometry, out_shape=dst.shape, transform=dst.transform, invert=True)
            dst.write(mask.astype('uint8'), 1)


def get_GRANULE_dir(SAFE_dir):
    """ Returns the path to the GRANULE directory in the SAFE directory """
    return SAFE_dir / "GRANULE" / os.listdir(SAFE_dir / "GRANULE")[0]

def get_SAFE_extent(SAFE_dir):
    """ Returns the extent of the SAFE directory in the form [ulx, lry, lrx, uly] """
    MTD_TL_xml_path = str(get_GRANULE_dir(SAFE_dir) / "MTD_TL.xml")
    root = ET.parse(MTD_TL_xml_path).getroot()
    tile_geocoding_el = root.findall(".//Tile_Geocoding")[0]
    size_el = tile_geocoding_el.findall(".//Size[@resolution='10']")[0]
    nrow = int(size_el.find('NROWS').text)
    ncol = int(size_el.find('NCOLS').text)
    geoposition_element = root.findall(".//Geoposition[@resolution='10']")[0]
    ulx = int(geoposition_element.find('ULX').text)
    uly = int(geoposition_element.find('ULY').text)
    lrx = ulx + ncol * 10
    lry = uly - nrow * 10
    return [ulx, lry, lrx, uly]

def get_region_of_interest(raster_label, raster_SAFE_dir):
    """ Returns the region of interest for a safe file for a corresponding label raster file 
    
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

    print(f'label_extent: ulx: {label_ulx}\tuly: {label_uly}\tlrx: {label_lrx}\tlry: {label_lry}')
    print(f'SAFE_extent: ulx:  {SAFE_ulx}\tuly: {SAFE_uly}\tlrx: {SAFE_lrx}\tlry: {SAFE_lry}')
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
    buffer = 12 # 12 pixel buffer around the roi

    nrow_win =  math.ceil(roi[2] / 6) * 6 + buffer # make sure nrow_win is divisible by 6
    ncol_win =  math.ceil(roi[3] / 6) * 6 + buffer

    row0 = roi[0] + math.floor(roi[2] / 2) # center of roi
    col0 = roi[1] + math.floor(roi[3] / 2)

    sen2cor_roi = [str(row0), str(col0), str(nrow_win), str(ncol_win)]

    return sen2cor_roi