# import pandas as pd
# import random
# from faker import Faker
# from datetime import datetime

# fake = Faker()

# # Load User and Subscription data
# users = pd.read_excel("data/raw_data/SubscriptionUseCase_Dataset.xlsx", sheet_name="User_Data")
# subs = pd.read_excel("data/raw_data/SubscriptionUseCase_Dataset.xlsx", sheet_name="Subscriptions")



# # Convert Status: Active → 1, Inactive → 0
# users['Status'] = users['Status'].map({"Active": 1, "Inactive": 0})

# # Ensure dates are datetime
# subs['Start Date'] = pd.to_datetime(subs['Start Date'])
# # subs['End Date'] = pd.to_datetime(subs['End Date'], errors="coerce")
# print(subs.columns.tolist())


# enriched = []

# for _, user in users.iterrows():
#     uid = user['User Id']
#     user_subs = subs[subs['User Id'] == uid].sort_values('Start Date')

#     # Current & previous products
#     current_prod = None
#     prev_prods = []
#     if not user_subs.empty:
#         current = user_subs[user_subs['Status'] == "Active"]
#         if not current.empty:
#             current_prod = current.iloc[-1]['Product Id']
#         prev_prods = user_subs['Product Id'].tolist()[:-1]

#         # Duration
#     #     latest = user_subs.iloc[-1]
#     #     if pd.notna(latest['End Date']):
#     #         duration = (latest['End Date'] - latest['Start Date']).days
#     #     else:
#     #         duration = (datetime.now() - latest['Start Date']).days
#         duration = 0
#     else:
#         duration = 0

#     # Synthetic demographics
#     age = random.randint(18, 60)
#     gender = random.choice(["Male", "Female", "Other"])
#     location = fake.city()

#     # Engagement time (simulate hrs per month)
#     engagement_time = random.randint(1, 50)

#     # Rating (simulate)
#     if "Inactive" in user_subs['Status'].values:
#         rating = random.choice([0, 2])
#     elif "Active" in user_subs['Status'].values:
#         rating = random.choice([4, 5])
#     else:
#         rating = 3

#     # Discount history (simulate 30% have discounts)
#     discount_history = random.choice([True, False, False])

#     enriched.append({
#         "User Id": uid,   # keep the same column name
#         "age": age,
#         "gender": gender,
#         "location": location,
#         "current_product_id": current_prod,
#         "previous_products": prev_prods,
#         "duration": duration,
#         "rating": rating,
#         "engagement_time": engagement_time,
#         "discount_history": discount_history
#     })

# # Convert to DataFrame
# enriched_df = pd.DataFrame(enriched)

# # Merge with User Data (keeps original names untouched)
# final_df = users.merge(enriched_df, on="User Id", how="left")

# # Save enriched dataset
# final_df.to_excel("Enriched_User_Data.xlsx", index=False)

# print("Enriched dataset saved with Status as 0/1 ✅")










import pandas as pd
import random
from faker import Faker
from datetime import datetime

fake = Faker()

# Load User and Subscription data
users = pd.read_excel("data/raw_data/SubscriptionUseCase_Dataset.xlsx", sheet_name="User_Data")
subs = pd.read_excel("data/raw_data/SubscriptionUseCase_Dataset.xlsx", sheet_name="Subscriptions")

# Convert Status: Active → 1, Inactive → 0
users['Status'] = users['Status'].map({"Active": 1, "Inactive": 0})

# Ensure dates are datetime
subs['Start Date'] = pd.to_datetime(subs['Start Date'], errors="coerce")
subs['Terminated Date'] = pd.to_datetime(subs['Terminated Date'], errors="coerce")

enriched = []

for _, user in users.iterrows():
    uid = user['User Id']
    user_subs = subs[subs['User Id'] == uid].sort_values('Start Date')

    # Current & previous products
    current_prod = None
    prev_prods = []
    if not user_subs.empty:
        current = user_subs[user_subs['Status'] == "Active"]
        if not current.empty:
            current_prod = current.iloc[-1]['Product Id']
        prev_prods = user_subs['Product Id'].tolist()[:-1]

        # Duration
        latest = user_subs.iloc[-1]
        if pd.notna(latest['Terminated Date']):
            duration = (latest['Terminated Date'] - latest['Start Date']).days
        else:
            duration = (datetime.now() - latest['Start Date']).days
    else:
        duration = 0

    # Synthetic demographics
    age = random.randint(18, 60)
    gender = random.choice(["Male", "Female", "Other"])
    location = fake.city()

    # Engagement time (simulate hrs per month)
    engagement_time = random.randint(1, 50)

    # Rating (simulate)
    if "Inactive" in user_subs['Status'].values:
        rating = random.choice([0, 2])
    elif "Active" in user_subs['Status'].values:
        rating = random.choice([4, 5])
    else:
        rating = 3

    # Discount history (simulate 30% have discounts)
    discount_history = random.choice([True, False, False])

    enriched.append({
        "User Id": uid,
        "age": age,
        "gender": gender,
        "location": location,
        "current_product_id": current_prod,
        "previous_products": prev_prods,
        "duration": duration,
        "rating": rating,
        "engagement_time": engagement_time,
        "discount_history": discount_history
    })

# Convert to DataFrame
enriched_df = pd.DataFrame(enriched)

# Merge with User Data (keep only required columns)
final_df = users.merge(enriched_df, on="User Id", how="left")

# Reorder columns to match your target dataset
final_df = final_df[[
    "User Id", "Name", "Phone", "Email", "Status",
    "age", "gender", "location",
    "current_product_id", "previous_products",
    "duration", "rating", "engagement_time", "discount_history"
]]

# Save enriched dataset
final_df.to_excel("Enriched_User_Data.xlsx", index=False)

print("Enriched dataset saved with required features ✅")
print(final_df.head())