import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math
import logging
import wrds
import matplotlib.pyplot as plt
import os

# -------------------------
# Logging Setup
# -------------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_file_logger(log_file):
    """Add a file handler to the logger."""
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

# Set up a file logger.
log_filename = "/Users/dkewlan/My WorkSpace/Final Project/training.log"
setup_file_logger(log_filename)
logger.info("File logger initialized.")

# -------------------------
# Establish WRDS Connection
# -------------------------
db = wrds.Connection()  # Ensure your WRDS credentials/environment are set up