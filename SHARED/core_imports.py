# shared_core.py

# === Standard Library ===
import os
import sys
import time
import math
import subprocess
import random
import logging
import traceback
import inspect
import io
import cProfile
import pstats
from typing import Union, Dict, Any, Optional
from importlib import util as importlib_util
from enum import Enum
from collections import namedtuple, defaultdict
import datetime


# === Third-Party Libraries ===
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import noise
from scipy.ndimage import gaussian_filter, label
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns
import matplotlib
import pandas as pd
import re

from tqdm import tqdm


import os
import subprocess
import webbrowser
import threading
import time

# === Utilities ===

from UTILITIES.utils_logger import Logger
from UTILITIES.utils_tensorboard import TensorBoardLogger

tensorboard_logger = TensorBoardLogger()
from UTILITIES.utils_matplot import MatplotlibPlotter

from UTILITIES.utils_helpers import (
    profile_function,
    find_closest_actor,
    generate_random_colour,
    cleanup,
)
