import os
import rasterio
import numpy as np
import xml.etree.ElementTree as ET

# hardcoded for pf-pc12
SEN2COR_HOME = "/scratch/toml/Sen2Cor-02.11.00-Linux64"

input_dir = "input"
output_dir = "output"


def modify_GIPP(reference_GIPP, custom_GIPP, region_of_interest=None):
    GIPP_tree = ET.parse(reference_GIPP)
    root = GIPP_tree.getroot()
    
    # set region of interest
    if region_of_interest is not None:
        root.find("Region_Of_Interest").find("row0").text = region_of_interest[0]
        root.find("Region_Of_Interest").find("col0").text = region_of_interest[1]
        root.find("Region_Of_Interest").find("nrow_win").text = region_of_interest[2]
        root.find("Region_Of_Interest").find("ncol_win").text = region_of_interest[3]

    # Modify Atmospheric_Correction.Look_Up_Tables for Aerosol_Type, Mid_Latitude, Ozone_Content
    # RURAL, MARITIME, AUTO 
    if os.environ.get("SEN2COR_MODIFY_LUT_AEROSOL") is not None:
        root.find("Atmospheric_Correction").find("Look_Up_Tables").find(
            "Aerosol_Type"
        ).text = os.environ.get("SEN2COR_MODIFY_LUT_AEROSOL")
    # SUMMER, WINTER, AUTO
    if os.environ.get("SEN2COR_MODIFY_LUT_SEASON") is not None:    
        root.find("Atmospheric_Correction").find("Look_Up_Tables").find(
            "Mid_Latitude"
        ).text = os.environ.get("SEN2COR_MODIFY_LUT_SEASON")
    #  see GIPP for explanation, default: 0
    if os.environ.get("SEN2COR_MODIFY_LUT_OZONE") is not None:    
        root.find("Atmospheric_Correction").find("Look_Up_Tables").find(
            "Ozone_Content"
        ).text = os.environ.get("SEN2COR_MODIFY_LUT_OZONE")
    
    # Modify Atmospheric_Correction.Flags for WV_Correction, Cirrus_Correction, BRDF_Correction
    # 0: No WV correction, 1: correction with 940 nm band (default)
    if os.environ.get("SEN2COR_MODIFY_FLAGS_WV") is not None:
        root.find("Atmospheric_Correction").find("Flags").find(
            "WV_Correction"
        ).text = os.environ.get("SEN2COR_MODIFY_FLAGS_WV")
    # FALSE: no cirrus correction applied, TRUE: cirrus correction applied
    if os.environ.get("SEN2COR_MODIFY_FLAGS_CIRRUS") is not None:
        root.find("Atmospheric_Correction").find("Flags").find(
            "Cirrus_Correction"
        ).text = os.environ.get("SEN2COR_MODIFY_FLAGS_CIRRUS")
    # 0: no BRDF correction, 1, 2: see IODD for explanation
    if os.environ.get("SEN2COR_MODIFY_FLAGS_BRDF") is not None:
        root.find("Atmospheric_Correction").find("Flags").find(
            "BRDF_Correction"
        ).text = os.environ.get("SEN2COR_MODIFY_FLAGS_BRDF")

    # Modify Atmospheric_Correction.Calibration for Visibility
    # visibility (5 <= visib <= 120 km), default: 40
    if os.environ.get("SEN2COR_MODIFY_CALIB_VISIBILITY") is not None:
        root.find("Atmospheric_Correction").find("Calibration").find(
            "Visibility"
        ).text = os.environ.get("SEN2COR_MODIFY_CALIB_VISIBILITY")
    
    # write changes to custom GIPP
    GIPP_tree.write(custom_GIPP)
    
    
class L2A_process_runner:
    """
    Helper class to run the L2A process
    """

    def __init__(self, input_dir, output_dir, l2a_process_loc=None, resolution=10):
        """
        variables:
            input_dir: directory containing the L1C products
            output_dir: directory where the L2A products will be stored
            l2a_process_loc: location of the L2A_Process executable

        """
        if l2a_process_loc is None:
            self.l2a_process_loc = SEN2COR_HOME + "/bin/L2A_Process"
        else:
            self.l2a_process_loc = l2a_process_loc
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.resolution = resolution

    def run(self, region_of_interest=None):
        """
        Run the L2A process
        """
        reference_GIPP =  "/home/toml/sen2cor/2.11/cfg/L2A_GIPP.xml"
        custom_GIPP = "/home/toml/sen2cor/2.11/cfg/L2A_GIPP_custom.xml"

        modify_GIPP(reference_GIPP, custom_GIPP, region_of_interest=region_of_interest)

        if os.environ.get("SEN2COR_MOD_SC_ONLY") is not None:
            sc_only = " --sc_only"
        else:
            sc_only = ""

        cmd = f"{self.l2a_process_loc} {self.input_dir} --resolution {self.resolution} --GIP_L2A {custom_GIPP} --output_dir {self.output_dir}{sc_only}"
        print("Running command: ", cmd)
        os.system(cmd)


if __name__ == "__main__":
    input_dir = "/scratch/toml/sentinel_2_data/S2A_MSIL1C_20230906T102601_N0509_R108_T32TMT_20230906T141124.SAFE"
    output_dir = "/scratch/toml/sentinel_2_data"

    os.environ["SEN2COR_MODIFY_SCL"] = "1"
    os.environ["SEN2COR_MODIFY_CLOUD_CLASSIFICATION"] = "1"

    l2a_process_runner = L2A_process_runner(
        input_dir, output_dir=output_dir, resolution=60
    )
    l2a_process_runner.run()
