"""
Simple client to authenticate, search and download sentinel data from the Copernicus Dataspace Ecosystem.
"""
import json
import requests
import pandas as pd


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
        filter_string = f"$filter=Collection/Name eq '{platform}' and ContentDate/Start gt {start_date}T00:00:00.000Z and ContentDate/Start lt {end_date}T00:00:00.000Z and OData.CSC.Intersects(area=geography'SRID=4326;{area_of_interest}')"
        request_url = f"{catalog_url}?{filter_string}"
        print(request_url)
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
        else:
            raise Exception("Download failed.")

