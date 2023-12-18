"""
Simple client to authenticate, search and download sentinel data from the Copernicus Dataspace Ecosystem.
"""
import pathlib
import json
import requests
import pandas as pd
import os
from osgeo import gdal



class Client(object):
    """
    Simple client to authenticate, search and download sentinel data from the Copernicus Dataspace Ecosystem.
    """

    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.base_url = "TODO"
        self.token = None
        self._authenticate()

    def _authenticate(self):
        """
        Set access token for the Copernicus Dataspace Ecosystem using the provided username and password.
        """
        url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        data = {
            "client_id": "cdse-public",
            "username": self.username,
            "password": self.password,
            "grant_type": "password",
        }
        try:
            response = requests.post(url, data=data)
            response.raise_for_status()
        except Exception as e:
            raise Exception(
                f"Access token creation failed. Reponse from the server was: {response.json()}"
            )
        self.token = response.json()["access_token"]

    def search(self, start_date, end_date, platform, area_of_interest):
        catalog_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
        filter_string = f"$filter=Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le 40.00) and Collection/Name eq '{platform}' and ContentDate/Start gt {start_date}T00:00:00.000Z and ContentDate/Start lt {end_date}T00:00:00.000Z and OData.CSC.Intersects(area=geography'SRID=4326;{area_of_interest}')"
        request_url = f"{catalog_url}?{filter_string}"
        try:
            response = requests.get(request_url)
            response.raise_for_status()
        except Exception as e:
            raise Exception(
                f"Search failed. Reponse from the server was: {response.json()}"
            )
        return pd.DataFrame.from_dict(response.json()["value"])

    def download(self, product_id, output_path):
        """
        Download data from the Copernicus Dataspace Ecosystem.
        """
        url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
        headers = {"Authorization": f"Bearer {self.token}"}
        response = requests.get(url, headers=headers, stream=True)
        if response.status_code == 200:
            with open(output_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
        elif response.status_code == 401:
            try:
                self._authenticate()
                self.download(product_id, output_path)
            except Exception as e:
                raise e
        else:
            print(response.status_code)
            raise Exception("Download failed.")


def get_labels(train_path):
    labels = []
    for train_dir in os.listdir(train_path):
        geojson_filename = (
            f"global_monthly_2020_01_mosaic_{train_dir}_Buildings.geojson"
        )
        label_geojson = pathlib.Path(train_path) / train_dir / "labels" / geojson_filename
        if os.path.isfile(label_geojson):
            labels.append(label_geojson)
        # else:
        #     print(f"Label file {label_geojson} not found")
    return labels


def get_sentinel2_product_ids(client: Client, labels):
    """Takes a list of geojson labels and returns a dictionary of product ids and labels."""

    start_date = "2020-01-01"
    end_date = "2020-01-31"
    platform = "SENTINEL-2"
    products = {}
    for label in labels:
        try:
            label_name = str(label).split("_mosaic_")[-1].split("_Buildings")[0]
            print(label_name, end=": ")
            ds = gdal.OpenEx(str(label), gdal.OF_VECTOR)
            extent = ds.GetLayer().GetExtent()
            footprint = f"POLYGON(({extent[0]} {extent[2]}, {extent[1]} {extent[2]}, {extent[1]} {extent[3]}, {extent[0]} {extent[3]}, {extent[0]} {extent[2]}))"
            prod_list = client.search(start_date,end_date,platform,footprint)
            for index, row in prod_list.iterrows():
                # check name contains "MSIL1C"
                if "MSIL1C" in row['Name']: # Only use L1C products
                    print(f"Found {row['Name']}")
                    products[label_name] = row.to_dict()
                    products[label_name]['geojson_path'] = label
                    break
        except Exception as e:
            print(f"Error: {e}")
            continue
            # raise e
    return products


def download_sentinel2_products(client: Client, products, output_path):
    """Takes a dictionary of product ids and labels and downloads the products."""
    downloaded_products = {}
    for label_name, product in products.items():
        print(f"Downloading {label_name}...")
        output_filename = pathlib.Path(output_path) / f"{label_name}.zip"
        product['zip_path'] = output_filename
        if os.path.isfile(output_filename):
            print(f"File {output_filename} already exists")
            continue
        client.download(product['Id'], output_filename)
    return products

