import os
import json

import rasterio as rio
import numpy as np

import l2a_runner


class L2A_Band(object):
    def __init__(self, jp2_path, name, band_id):
        self.jp2_path = jp2_path
        self.name = name
        self.id = band_id
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

    def __str__(self) -> str:
        return object.__str__(self) + f" jp2_path: {self.jp2_path}"
    
class L2A_Band_Stack(object):
    def __init__(self, granule_image_data_path):
        self.band_names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B8A',
                           'B09', 'B11', 'B12', 'AOT', 'SCL', 'TCI', 'WVP']
        self.data_path = granule_image_data_path
        self.bands = {}
        self.meta = {}
        self.init_bands()
    
    # allows stack to be indexed like a dictionary
    def __getitem__(self, key):
        return self.bands[key]
    
    def init_bands(self):
        for band in os.listdir(self.data_path):
            band_path = f"{self.data_path}/{band}"
            band_name = band.split("_")[2]
            if band_name in self.band_names:
                idx = self.band_names.index(band_name)
                self.bands[band_name] = L2A_Band(band_path, band_name, idx)
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
        Valid band_names arguments:
            None: all bands incl meta bands
            'rgb': bands B04, B03, B02
            'false_color': bands B8A, B04, B03
            'bands': all bands excl meta bands
            list of band names: bands with names in list
        """
        if band_names is None:
            band_names = self.band_names
        elif band_names == "rgb":
            band_names = ['B04', 'B03', 'B02']
        elif band_names == "false_color":
            band_names = ['B8A', 'B04', 'B03']
        elif band_names == "bands":
            band_names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B8A', 'B09', 'B11', 'B12']
        elif isinstance(band_names, list):
            # check if all band names are valid
            for band_name in band_names:
                if band_name not in self.band_names:
                    raise ValueError(f"Invalid band name: {band_name}, expected one of {self.band_names}")
            band_names = [band_names]
        else:
            raise ValueError(f"Invalid band_names argument: {band_names}, expected None, 'rgb', 'false_color', or list of band names")

        return [self.bands[band].read() for band in band_names]
    
    
    def create_new_band(self, band_name: str, data: np.ndarray):
        """
        Create a new band in the band stack
        Arguments:
            band_name: string of band name
            data: numpy array of band data
        """
        if band_name in self.band_names:
            raise ValueError(f"Band {band_name} already exists in band stack")
        else:
            self.band_names.append(band_name)
            example_name = os.listdir(self.data_path)[0].split("_")
            file_name = f"{example_name[0]}_{example_name[1]}_{band_name}.jp2"
            band_meta = self.meta.copy()
            band_meta['count'] = 1
            with rio.open(f"{self.data_path}/{file_name}", 'w', **self.meta) as dst:
                dst.write(data, 1)
            self.bands[band_name] = L2A_Band(f"{self.data_path}/{file_name}", band_name)
            self.meta['count'] += 1




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
            self.modified_bands[loc_name] = {}

        # populate reference dict
        for data_run in self.data_info["reference"]:
            loc = data_run["loc"]
            loc_name = loc["loc_name"]
            granule_path = f"{data_run['output_dir']}/{os.listdir(data_run['output_dir'])[0]}/GRANULE"
            granule_path = f"{granule_path}/{os.listdir(granule_path)[0]}"
            jp2_path = f"{granule_path}/IMG_DATA/R{self.resolution}m/"

            self.reference_bands[loc_name] = L2A_Band_Stack(jp2_path)

        # populate modified dict
        for data_run in self.data_info["modified"]:
            loc = data_run["loc"]
            loc_name = loc["loc_name"]
            mod_name = data_run["mod"]["name"]
            
            granule_path = f"{data_run['output_dir']}/{os.listdir(data_run['output_dir'])[0]}/GRANULE"
            granule_path = f"{granule_path}/{os.listdir(granule_path)[0]}"
            jp2_path = f"{granule_path}/IMG_DATA/R{self.resolution}m/"
            
            self.modified_bands[loc_name][mod_name] = L2A_Band_Stack(jp2_path)


    def clean_l2a_data(self):
        self.reference_bands = {}
        self.modified_bands = {}
        # recursively delete report_dir
        os.system(f"rm -r {self.report_dir}")
