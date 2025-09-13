import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Get the directory where the script is located
script_dir = os.path.dirname(__file__)

# Paths for the datasets, relative to the script's directory
RAW_FILE = os.path.join(script_dir, "data", "raw_data", "Enriched_User_Data.xlsx")
TRAIN_DIR = os.path.join(script_dir, "data", "train_data")
TEST_DIR = os.path.join(script_dir, "data", "test_data")

# Loading data
df = pd.read_excel(RAW_FILE)

# The 'Status' column contains only NaN values.
# We will fill these NaN values with a default value, for example, 1 (for 'Active').
# This is a necessary step to make the data usable for training.
df['Status'].fillna(1, inplace=True)

# Ensure the column is of integer type after filling
df['Status'] = df['Status'].astype(int)

# Fill null 'current_product_id' based on 'previous_products'
def fill_current_product(row):
    # Safely evaluate the string representation of the list
    try:
        prev_products = eval(row['previous_products'])
        if isinstance(prev_products, list) and prev_products:
            return prev_products[-1] # Get the last product
    except (SyntaxError, NameError):
        # Handle cases where the string is not a valid list
        pass
    return 0 # Default value if previous_products is empty or invalid

df['current_product_id'] = df.apply(fill_current_product, axis=1)


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
