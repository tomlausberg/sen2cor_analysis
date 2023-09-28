import os
import rasterio
import numpy as np

# hardcoded for pf-pc12
SEN2COR_HOME = "/scratch/toml/Sen2Cor-02.11.00-Linux64"

input_dir = "input"
output_dir = "output"

class L2A_process_runner():
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

    def run(self):
        """ 
        Run the L2A process
        """
        # cmd = self.l2a_process_loc + " --resolution 60 " + self.input_dir + " --output_dir " + self.output_dir
        cmd = f"{self.l2a_process_loc} {self.input_dir} --resolution {self.resolution} --output_dir {self.output_dir}"
        os.system(cmd)

if __name__ == "__main__":
    input_dir = "/scratch/toml/sentinel_2_data/S2A_MSIL1C_20230906T102601_N0509_R108_T32TMT_20230906T141124.SAFE"
    output_dir = "/scratch/toml/sentinel_2_data"

    os.environ["SEN2COR_MODIFY_SCL"] = "1"
    os.environ["SEN2COR_MODIFY_CLOUD_CLASSIFICATION"] = "1"

    l2a_process_runner = L2A_process_runner(input_dir, output_dir=output_dir, resolution=60)
    l2a_process_runner.run()

