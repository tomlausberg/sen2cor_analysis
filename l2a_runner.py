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

def scale_SCL(image_file):
    modified_image_file = image_file.split(".")[0] + "_modified.jp2"
    SCL = rasterio.open(image_file)
    data = SCL.read()
    scaling_factor = 7 # ~ 255/33
    data = data * scaling_factor
    with rasterio.open('modified_image_file.jp2', 'w', **SCL.meta) as dst:
        dst.write(data)


def scl_to_rgb(rio_scl):
    # map the data to the following colors
    colour_dict = {
        0: (0, 0, 0),  # black: no data
        1: (255,0,0),  # red: saturated or defective
        2: (50,50,50),  # dark grey: casted shadows
        3: (165,42,42),  # brown: cloud shadows
        4: (0,100,0),  # green: vegetation
        5: (255,255,0),  # yellow: bare soils
        6: (0,0,255),  # blue: water
        7: (128,128,128),  # grey: unclassified
        8: (211,211,211),  # light grey: cloud medium probability
        9: (255,255,255),  # white: cloud high probability
        10: (0,255,255),  # light blue: thin cirrus
        11: (255,192,203)  # pink: snow
    }

    # create a new array with the same shape as the original data
    # but with 3 channels (RGB)
    scl_array = rio_scl.read(1)
    rgb = np.zeros((rio_scl.height, rio_scl.width, 3), dtype=np.uint8)

    # map the data to RGB
    for k, v in colour_dict.items():
        rgb[scl_array == k] = v

    # plot the RGB image
    import matplotlib.pyplot as plt
    plt.imshow(rgb)
    plt.show()

if __name__ == "__main__":
    input_dir = "/scratch/toml/sentinel_2_data/S2A_MSIL1C_20230906T102601_N0509_R108_T32TMT_20230906T141124.SAFE"
    output_dir = "/scratch/toml/sentinel_2_data"

    os.environ["SEN2COR_MODIFY_SCL"] = "1"
    os.environ["SEN2COR_MODIFY_CLOUD_CLASSIFICATION"] = "1"

    l2a_process_runner = L2A_process_runner(input_dir, output_dir=output_dir, resolution=60)
    l2a_process_runner.run()

