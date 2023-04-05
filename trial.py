# Download the dataset
import os
from pathlib import Path

if not os.path.exists('KTH_pickle.zip'):
  wget --no-check-certificate https://empslocal.ex.ac.uk/people/staff/ad735/ECMM426/KTH_pickle.zip
  unzip -q KTH_pickle.zip

# Dictionary of categories
CATEGORY_INDEX = {
    "boxing": 0,
    "handclapping": 1,
    "handwaving": 2,
    "jogging": 3,
    "running": 4,
    "walking": 5
}

dir_pickle = Path('KTH_pickle/')
filepath = os.path.join(dir_pickle, "train.pickle")