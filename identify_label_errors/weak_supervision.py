####################################################
#  Use labeling functions to weakly supervise the 
#  machine learning problem
####################################################

import pandas as pd
import numpy as np
from .utils import Config

args = Config(config_file_path='config.yaml').parse()