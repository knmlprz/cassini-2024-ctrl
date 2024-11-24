#!/usr/bin/env python3

import subprocess
import kagglehub


# Download latest version of wildfire prediction dataset
path = kagglehub.dataset_download("abdelghaniaaba/wildfire-prediction-dataset")
subprocess.run(["mv", path, "./data/"])
