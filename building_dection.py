import os
import rasterio as rio

import numpy as np
import json
import pyproj
from osgeo import gdal
import subprocess
import zipfile
from datetime import datetime

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
            f"global_monthly_2018_01_mosaic_{train_dir}_Buildings.geojson"
        )
        label_geojson = f"{train_path}/{train_dir}/labels/{geojson_filename}"
        if os.path.isfile(label_geojson):
            labels.append(label_geojson)
        # else:
        #     print(f"Label file {label_geojson} not found")
    return labels


def get_sentinel2_product_id(footprint, date=("20200101", "20200131")):
    """Takes a WKT footprint and returns the product id of the least cloudy Sentinel-2 product or None if no product is found."""
    user, password = get_credentials()
    api = SentinelAPI(user, password, "https://apihub.copernicus.eu/apihub/")

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
