import os
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import time
import zipfile

"""
Geospatial Data Download and Extraction Module

Key Features:
- Automated web scraping for shapefile ZIP downloads using Selenium
- Batch download capability for multiple geographic datasets
- Parallel extraction of compressed archives
- Duplicate file detection and handling
- Headless browser operation support

Dependencies:
- selenium: Web browser automation for dynamic content scraping
- requests: HTTP library for file download operations  
- zipfile: Standard library for archive extraction
"""

def download_zip(chromedriver_path, target_url, download_dir):
    """
    Automated web scraping and download function for geospatial ZIP files.
    """
    # Configuration and initialization
    options = Options()
    options.add_argument("--headless")

    service = Service(executable_path=chromedriver_path)
    driver = webdriver.Chrome(service=service)

    # Navigate to target URL and wait for page load
    driver.get(target_url)
    time.sleep(5)

    # Collect all shapefile ZIP download links containing "latest" version
    zip_links = []
    elements = driver.find_elements(By.TAG_NAME, 'a')
    for element in elements:
        href = element.get_attribute('href')
        if href and href.endswith('.shp.zip'):
            if "latest" in href.split('-'):
                zip_links.append(href)
    
    # Download each identified ZIP file
    for zip_url in zip_links:
        print(f"Downlaoding {zip_url} ...")

        # Construct local filename from URL
        zip_name = os.path.join(download_dir, zip_url.split('/')[-1])

        if os.path.exists(zip_name):
            print(f"File {zip_name} already exists. Skipping download...\n")
            continue
        
        # Stream download with chunked writing for memory efficiency
        with requests.get(zip_url, stream=True) as r:
            with open(zip_name, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        print(f"Finish downlaoding {zip_name}\n")
    
    print("Finish downloading all .ship.zip files !!!")

def extract_zips(zip_files_path, shp_files_path):
    """
    Batch extraction utility for downloaded geospatial ZIP archives.
    """
    def extract_one_zip(zip_file):
        # Construct full paths for source and destination
        zip_path = os.path.join(zip_files_path, zip_file)
        unzip_path = os.path.join(shp_files_path, os.path.splitext(zip_file)[0])
        

        if os.path.exists(unzip_path):
            print(f"{zip_file} has already been extracted.")
            return

        # Create extraction directory and extract archive contents
        os.makedirs(unzip_path, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            print(f"Extracting {zip_file}")
            zip_ref.extractall(unzip_path)
        
        print(f"Extracted {zip_file} successfully!\n")
    
    zip_files = [f for f in os.listdir(zip_files_path) if f.endswith('.zip')]

    for file in zip_files:
        extract_one_zip(file)

    print("All ZIP files have been extracted.")


    