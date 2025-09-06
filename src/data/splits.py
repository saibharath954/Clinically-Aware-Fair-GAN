import pandas as pd
from sklearn.model_selection import train_test_split
import os

# --- Configuration ---
INPUT_CSV = 'data/splits/master_subset_2k.csv'
OUTPUT_DIR = 'data/splits/'
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15 # Should sum to 1.0 with the others

# We stratify by both the target label and the sensitive attribute
STRATIFY_COLS = ['Pneumonia', 'race_group']
RANDOM_STATE = 42

def create_splits():
    """
    Loads the master dataframe and creates stratified train, validation,
    and test CSV files.
    """
    print(f"Loading data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    # Create a combined stratification column to handle multiple columns
    df['stratify_key'] = df[STRATIFY_COLS].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

    # First split: separate out the training set
    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - TRAIN_SIZE),
        random_state=RANDOM_STATE,
        stratify=df['stratify_key']
    )

    # Second split: split the remainder into validation and test sets
    # The new test_size is relative to the size of temp_df
    relative_test_size = TEST_SIZE / (VAL_SIZE + TEST_SIZE)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        random_state=RANDOM_STATE,
        stratify=temp_df['stratify_key']
    )

    # Drop the temporary stratification key
    train_df = train_df.drop(columns=['stratify_key'])
    val_df = val_df.drop(columns=['stratify_key'])
    test_df = test_df.drop(columns=['stratify_key'])

    # --- Save the splits ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_path = os.path.join(OUTPUT_DIR, 'train.csv')
    val_path = os.path.join(OUTPUT_DIR, 'val.csv')
    test_path = os.path.join(OUTPUT_DIR, 'test.csv')

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\n--- Split Summary ---")
    print(f"Total samples: {len(df)}")
    print(f"Training samples: {len(train_df)} ({len(train_df)/len(df):.2%})")
    print(f"Validation samples: {len(val_df)} ({len(val_df)/len(df):.2%})")
    print(f"Test samples: {len(test_df)} ({len(test_df)/len(df):.2%})")
    print("\n✅ Splits created successfully!")
    print(f"Files saved to: {train_path}, {val_path}, {test_path}")

if __name__ == '__main__':
    create_splits()