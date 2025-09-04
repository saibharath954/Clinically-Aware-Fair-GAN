# src/data/preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# --- 1. DEFINE FILE PATHS ---
# Make sure these paths are correct relative to your project's root directory
ADMISSIONS_CSV = 'data/admissions.csv'
PATIENTS_CSV = 'data/patients.csv'
RECORDS_CSV = 'data/cxr-record-list.csv'
CHEXPERT_CSV = 'data/mimic-cxr-2.0.0-chexpert.csv'
OUTPUT_CSV = 'data/splits/master_subset_2k.csv'

TARGET_SIZE = 2000

print("--- Starting Data Preprocessing ---")

# --- 2. LOAD ALL DATASETS ---
try:
    admissions_df = pd.read_csv(ADMISSIONS_CSV) 
    patients_df = pd.read_csv(PATIENTS_CSV)
    records_df = pd.read_csv(RECORDS_CSV)
    chexpert_df = pd.read_csv(CHEXPERT_CSV)
    print("✅ All CSV files loaded successfully.")
    print(f"Admissions data shape: {admissions_df.shape}")
    print(f"Patients data shape: {patients_df.shape}")
    print(f"Records data shape: {records_df.shape}")
    print(f"CheXpert labels shape: {chexpert_df.shape}")
except FileNotFoundError as e:
    print(f"❌ Error loading files: {e}")
    print("Please ensure all required CSV files are in the 'data/' directory.")
    exit()

# --- 3. PREPROCESS AND CLEAN DATA ---

# Clean CheXpert labels: We need a binary label for Pneumonia.
# The labeler uses 1.0 (Positive), -1.0 (Uncertain), 0.0 (Negative).
# We will map Positive to 1 and treat Uncertain/Negative as 0.
chexpert_df['Pneumonia'] = chexpert_df['Pneumonia'].replace(-1.0, 0.0).fillna(0).astype(int)

# Clean Race column: The 'race' column is very detailed. For a fairness
# analysis with a small dataset, it's better to group them into broader categories.
def group_race(race_str):
    race_str = str(race_str).upper()
    if 'WHITE' in race_str:
        return 'WHITE'
    elif 'BLACK' in race_str:
        return 'BLACK'
    elif 'ASIAN' in race_str:
        return 'ASIAN'
    elif 'HISPANIC' in race_str:
        return 'HISPANIC/LATINO'
    else:
        return 'OTHER'

admissions_df['race_group'] = admissions_df['race'].apply(group_race)

# A patient might have multiple admissions with different races.
# We'll select the most frequent race for each subject_id.
race_counts = admissions_df.groupby('subject_id')['race_group'].apply(lambda x: x.mode()[0]).reset_index()
race_counts.rename(columns={'race_group': 'most_frequent_race'}, inplace=True)
print("\n✅ Data preprocessing and cleaning complete.")
print("Value counts for new 'most_frequent_race' column:")
print(race_counts['most_frequent_race'].value_counts())

# --- 4. MERGE THE DATAFRAMES ---

# Start with the core image-to-study mapping
# The most robust way to merge is to join on all common columns at once.
# 'records_df' and 'chexpert_df' share 'subject_id' and 'study_id'.
merged_df = pd.merge(records_df, chexpert_df, on=['subject_id', 'study_id'], how='inner')

# Add patient demographics.
# Merge with the most frequent race from admissions
final_df = pd.merge(merged_df, race_counts, on='subject_id', how='left')

# Merge with patient info (gender, anchor_age) from patients.csv
final_df = pd.merge(final_df, patients_df, on='subject_id', how='left')

# Drop any rows where race or gender info is missing after the joins.
final_df.dropna(subset=['most_frequent_race', 'gender', 'anchor_age'], inplace=True)

print(f"\n✅ Merging complete. Final combined shape: {final_df.shape}")
print("Columns in final DataFrame:", final_df.columns.tolist())

# --- 5. PERFORM STRATIFIED SAMPLING ---

# Stratified sampling is crucial for small datasets to ensure that the
# proportions of key groups (like race and disease status) are preserved.
# We create a temporary key to stratify on both columns simultaneously.
final_df['stratify_key'] = final_df['most_frequent_race'] + '_' + final_df['Pneumonia'].astype(str)

# Use train_test_split to perform the sampling. We only care about the 'train' part.
# The 'random_state' ensures that the sample is the same every time you run the script.
subset_df, _ = train_test_split(
    final_df,
    train_size=TARGET_SIZE,
    stratify=final_df['stratify_key'],
    random_state=42
)

print(f"\n✅ Stratified sampling complete. Subset size: {subset_df.shape}")

# --- 6. SAVE THE FINAL SUBSET ---

# Drop the temporary stratification key before saving
subset_df = subset_df.drop(columns=['stratify_key'])

# Select and reorder columns for clarity
final_columns = [
    'subject_id', 'study_id', 'dicom_id',
    'Pneumonia', 'most_frequent_race', 'gender', 'anchor_age'
]
final_subset_df = subset_df[final_columns]

# Rename 'most_frequent_race' to 'race_group' for simplicity in later steps
final_subset_df = final_subset_df.rename(columns={'most_frequent_race': 'race_group'})

final_subset_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n🎉 Success! Master subset file saved to: {OUTPUT_CSV}")
print("\nFirst 5 rows of your new dataset:")
print(final_subset_df.head())
