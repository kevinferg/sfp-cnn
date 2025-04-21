import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
from download_datasets import *
import os

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    url = "https://drive.google.com/file/d/1vJ-FTrL-gBp4kY_L-1oRhxe4sYf9Uxd1"
    data_folder = "./"
    download_and_extract_google_drive_zip(url, data_folder)