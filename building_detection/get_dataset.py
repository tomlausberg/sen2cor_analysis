#!/home/toml/semester_thesis/pipeline/venv/bin/python

import os
import json

from datetime import datetime

import building_dection as bd

sn7_path = "/scratch/toml/sn7"
building_label_geojsons = os.path.join(sn7_path, "SN7_buildings_train/train")
sentinel_data_path = os.path.join(sn7_path, "sentinel2")

num_data = 30

labels = bd.get_labels(building_label_geojsons)[:num_data]
# find the sentinel2 products for the
products = bd.get_sentinel2_product_ids(labels)

downloaded_prods = bd.get_sentinel2_products(products, download_path=sentinel_data_path)

now = datetime.now().strftime("%Y%m%d_%H%M%S")
info_json_fn = f"{sentinel_data_path}/product_infos_{now}.json"
with open(info_json_fn, "w") as f:
    json.dump(downloaded_prods, f)
