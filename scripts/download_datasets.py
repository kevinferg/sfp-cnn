import os
import gdown
import zipfile

def download_and_extract_google_drive_zip(url, folder):
    os.makedirs(folder, exist_ok=True)
    file_id = url.split("/")[-1]
    direct_download_url = f"https://drive.google.com/uc?id={file_id}"

    zip_file_path = os.path.join(folder, f"temp-{file_id}.zip")

    print(f"Downloading zip to: {zip_file_path}")
    gdown.download(direct_download_url, zip_file_path, quiet=False)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            file_name = os.path.basename(file_info.filename)
            extracted_file_path = os.path.join(folder, file_name)
            
            if not os.path.exists(extracted_file_path):
                print(f"Extracting {extracted_file_path}")
                with open(extracted_file_path, 'wb') as extracted_file:
                    extracted_file.write(zip_ref.read(file_info))
            else:
                print(f"{extracted_file_path} already exists. Skipping extraction of this file.")

    print(f"Removing {zip_file_path}")
    os.remove(zip_file_path)

if __name__ == "__main__":
    url = "https://drive.google.com/file/d/1mbKgWmByB4Pt6X2SUlHAnIpUouMwO_ld"
    data_folder = "./data"
    download_and_extract_google_drive_zip(url, data_folder)