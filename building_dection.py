import os
import rasterio as rio
import numpy as np
import json
import pyproj
from osgeo import gdal
import subprocess


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
    print(vector_crs)
    print(vector_extent)
    ds = gdal.Open(raster_fn)
    raster_crs = pyproj.CRS.from_wkt(ds.GetProjection())
    print(raster_crs)
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
    print(raster_extent)
    # crop raster to extent of geojson
    gdal.Translate(out_fn, ds, projWin=raster_extent)
