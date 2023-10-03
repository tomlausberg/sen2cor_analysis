import os
import json

import rasterio as rio
import numpy as np

import l2a_runner


class L2A_Band(object):
    def __init__(self, jp2_path):
        self.jp2_path = jp2_path
        with rio.open(self.jp2_path, driver="JP2OpenJPEG") as src:
            self.meta = src.meta
            # set attributes from meta to imitate rasterio datasetReader
            for key, value in self.meta.items():
                setattr(self, key, value)

    def read(self, number=1):
        data = None
        with rio.open(self.jp2_path, driver="JP2OpenJPEG") as src:
            data = src.read(number)
        return data
    
class L2A_Band_Stack(object):
    def __init__(self, granule_image_data_path):
        self.band_names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B8A',
                           'B09', 'B11', 'B12', 'AOT', 'SCL', 'TCI', 'WVP']
        self.data_path = granule_image_data_path
        self.bands = {}
        self.meta = {}
        self.init_bands()
    
    def init_bands(self):
        for band in os.listdir(self.data_path):
            # example band name: T32UNU_20190623T100031_B01_60m.jp2
            band_path = f"{self.data_path}/{band}"
            band_name = band.split("_")[2]
            if band_name in self.band_names:
                self.bands[band_name] = L2A_Band(band_path)
            else:
                raise ValueError(f"Invalid band name: {band_name}, expected one of {self.band_names}")
            
            # if first band, set meta
            if len(self.meta) == 0:
                self.meta = self.bands[band_name].meta
        
        self.meta['count'] = len(self.bands)

    def read(self, band_name):
        return self.bands[band_name].read()
    
    def read_bands(self, band_names=None):
        """
        Read multiple bands from the stack
        """
        if band_names is None:
            band_names = self.band_names
        elif band_names is "rgb":
            band_names = ['B04', 'B03', 'B02']
        elif band_names is "false_color":
            band_names = ['B08', 'B04', 'B03']
        elif isinstance(band_names, list):
            # check if all band names are valid
            for band_name in band_names:
                if band_name not in self.band_names:
                    raise ValueError(f"Invalid band name: {band_name}, expected one of {self.band_names}")
            band_names = [band_names]
        else:
            raise ValueError(f"Invalid band_names argument: {band_names}, expected None, 'rgb', 'false_color', or list of band names")

        return [self.bands[band].read() for band in band_names]
    
    
    




class L2A_Analysis(object):
    """
    Runs l2a process for different locations and locations.
    """

    def __init__(self, report_name="TEST"):
        self.base_input_dir = "/scratch/toml/sentinel_2_data/locations"
        self.base_output_dir = "/scratch/toml/sentinel_2_data/reports"
        self.report_dir = f"{self.base_output_dir}/{report_name}"
        self.resolution = 60
        self.bands = [
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B8A",
            "B09",
            "B11",
            "B12",
            "AOT",
            "SCL",
            "TCI",
            "WVP",
        ]

    def set_locations(self, locs):
        """
        Set locations from a dictionary
        locations: dictionary of locations in the form of {location_name: location_path}
        """
        self.locations = {}
        for loc_name, loc_path in locs.items():
            loc_product_description = loc_path.split(".")
            if loc_product_description[-1] != "SAFE":
                raise ValueError(
                    f"Invalid product description: {loc_product_description}. Expected SAFE file"
                )
            (
                misssion_id,
                product_level,
                date_take,
                processing_baseline,
                orbit_number,
                tile,
                product_discriminator,
            ) = loc_product_description[0].split("_")

            self.locations[loc_name] = {
                "loc_name": loc_name,
                "l1c_product_name": loc_product_description[0],
                "l1c_path": f"{self.base_input_dir}/{loc_name}/{loc_path}",
                "mission_id": misssion_id,
                "date_take": date_take,
                "processing_baseline": processing_baseline,
                "orbit_number": orbit_number,
                "tile": tile,
                "product_discriminator": product_discriminator.split(".")[0],
            }

    def set_modifications(self, modifications):
        """
        Set modifications from a list of dictionaries
        Dictionary format:
        {
            "flag": "environment variable to set",
            "value": "value to set the environment variable to"
            "info": "information about the modification"
            "name": "name of the modification"
        }
        """
        self.modifications = modifications

    def run_sen2cor(self):
        """
        Run the l2a process for each location and modification
        """
        self.data_info = {}
        # run reference sen2cor without modifications
        self.data_info["reference"] = []
        for loc_name, loc_dict in self.locations.items():
            mod_name = "reference"
            output_dir = f"{self.report_dir}/{loc_name}/{mod_name}"

            if os.path.isdir(output_dir):
                print(f"{loc_name}:\t{mod_name} already exists. Skipping...")
            else:
                print(f"Running sen2cor for {loc_name}: Reference")
                os.makedirs(output_dir, exist_ok=True)

                runner = l2a_runner.L2A_process_runner(
                    loc_dict["l1c_path"], output_dir, resolution=self.resolution
                )
                runner.run()

            info_dict = {
                "name": f"{loc_name}_{mod_name}",
                "loc": loc_dict,
                "mod": None,
                "output_dir": output_dir,
            }
            self.data_info["reference"].append(info_dict)

        # run sen2cor for each modification
        self.data_info["modified"] = []

        for mod in self.modifications:
            for loc_name, loc_dict in self.locations.items():
                mod_flag = mod["flag"]
                mod_val = mod["value"]
                mod_name = mod["name"]

                output_dir = f"{self.report_dir}/{loc_name}/{mod_name}"
                if os.path.isdir(output_dir):
                    print(f"{loc_name}:\t{mod_name} already exists. Skipping...")
                else:
                    print(f"Running sen2cor for {loc_name}: {mod_name}")
                    os.makedirs(output_dir, exist_ok=True)

                    os.environ[mod_flag] = mod_val
                    runner = l2a_runner.L2A_process_runner(
                        loc_dict["l1c_path"], output_dir, resolution=self.resolution
                    )
                    runner.run()
                    os.environ.pop(mod_flag, None)

                info_dict = {
                    "name": f"{loc_name}_{mod_name}",
                    "loc": loc_dict,
                    "mod": mod,
                    "output_dir": output_dir,
                }
                self.data_info["modified"].append(info_dict)

        with open(f"{self.report_dir}/data_info.json", "w") as f:
            json.dump(self.data_info, f, indent=4)

    def read_l2a_data(self):
        # initialize data dictionaries
        self.reference_bands = {}
        self.modified_bands = {}
        for loc_name, loc_dict in self.locations.items():
            self.reference_bands[loc_name] = {}
            self.modified_bands[loc_name] = {}
            for mod in self.modifications:
                self.modified_bands[loc_name][mod["name"]] = {}

        # populate reference dict
        for data_run in self.data_info["reference"]:
            loc = data_run["loc"]
            loc_name = loc["loc_name"]
            granule_path = f"{data_run['output_dir']}/{os.listdir(data_run['output_dir'])[0]}/GRANULE"
            granule_path = f"{granule_path}/{os.listdir(granule_path)[0]}"
            for band in self.bands:
                band_file = (
                    f"{loc['tile']}_{loc['date_take']}_{band}_{self.resolution}m.jp2"
                )
                jp2_path = f"{granule_path}/IMG_DATA/R{self.resolution}m/{band_file}"
                self.reference_bands[loc_name][band] = L2A_Band(jp2_path)

        # populate modified dict
        for data_run in self.data_info["modified"]:
            loc = data_run["loc"]
            loc_name = loc["loc_name"]
            mod_name = data_run["mod"]["name"]
            granule_path = f"{data_run['output_dir']}/{os.listdir(data_run['output_dir'])[0]}/GRANULE"
            granule_path = f"{granule_path}/{os.listdir(granule_path)[0]}"
            for band in self.bands:
                band_file = (
                    f"{loc['tile']}_{loc['date_take']}_{band}_{self.resolution}m.jp2"
                )
                jp2_path = f"{granule_path}/IMG_DATA/R{self.resolution}m/{band_file}"
                self.modified_bands[loc_name][mod_name][band] = L2A_Band(jp2_path)

    def clean_l2a_data(self):
        self.reference_bands = {}
        self.modified_bands = {}
        # recursively delete report_dir
        os.system(f"rm -r {self.report_dir}")
