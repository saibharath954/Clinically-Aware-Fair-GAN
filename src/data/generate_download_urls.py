import pandas as pd
import os

# Define file paths
SUBSET_CSV = 'data/splits/master_subset_2k.csv'
OUTPUT_TXT = 'data/files_to_download.txt'
BASE_URL = 'https://physionet.org/files/mimic-cxr-jpg/2.0.0/files/'

# --- 1. Load the subset CSV ---
try:
    df = pd.read_csv(SUBSET_CSV)
    print(f"✅ Successfully loaded {len(df)} images from {SUBSET_CSV}.")
except FileNotFoundError:
    print(f"❌ Error: {SUBSET_CSV} not found. Please run the preprocessing script first.")
    exit()

# --- 2. Generate the URLs ---
urls = []
for index, row in df.iterrows():
    # Construct the file path using the folder structure from PhysioNet
    # pXX/pXXXXXXXX/sXXXXXXXX/XXXXXXXX.jpg
    subject_id = str(row['subject_id'])
    study_id = str(row['study_id'])
    dicom_id = str(row['dicom_id'])

    # The first part of the patient folder is p and the first two digits of the subject_id
    subject_folder = 'p' + subject_id[:2]
    
    # The full path to the image
    image_path = os.path.join(subject_folder, f"p{subject_id}", f"s{study_id}", f"{dicom_id}.jpg")
    
    # Construct the final URL
    url = os.path.join(BASE_URL, image_path)
    urls.append(url)

# --- 3. Save URLs to a text file ---
# Create the parent directory if it doesn't exist
output_dir = os.path.dirname(OUTPUT_TXT)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(OUTPUT_TXT, 'w') as f:
    for url in urls:
        f.write(url + '\n')

print(f"🎉 Successfully created {OUTPUT_TXT} with {len(urls)} download URLs.")
print("The file is ready for use with the download_images.sh script.")