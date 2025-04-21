import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
import sys
sys.path.append('../scripts')
from download_datasets import *

if __name__ == "__main__":
    url = "https://drive.google.com/file/d/1vJ-FTrL-gBp4kY_L-1oRhxe4sYf9Uxd1"
    data_folder = "./"
    download_and_extract_google_drive_zip(url, data_folder)