import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Paths for the datasets
RAW_FILE = r"ml_model/data/raw_data/SubscriptionUseCase_Dataset.xlsx"
TRAIN_DIR = r"ml_model/data/train_data/"
TEST_DIR = r"ml_model/data/test_data/"

# Loading data
df = pd.read_excel(RAW_FILE)

# splitting data into 8:2 ratio
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Displaying the data size
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# Ensures those directories exist
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)


train_file = os.path.join(TRAIN_DIR, "train_subscriptions.json")
test_file = os.path.join(TEST_DIR, "test_subscriptions.json")

train_df.to_json(train_file, orient="records", lines=True)
test_df.to_json(test_file, orient="records", lines=True)

print(f"Train data saved to {train_file}")
print(f"Test data saved to {test_file}")
