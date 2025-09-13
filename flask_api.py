from flask import Flask, request, jsonify
import pandas as pd
import pickle
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# -------------------------
# MongoDB Connection
# -------------------------
mongodb_url = os.getenv("MONGODB_URL")
client = MongoClient(mongodb_url)
db = client.get_default_database()
users_collection = db["users"]
discounts_collection = db["discounts"]

# -------------------------
# Load ML Model
# -------------------------
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None


# ======================
# USER ENDPOINTS
# ======================
@app.route("/user/recommendations", methods=["POST"])
def user_recommendations():
    """
    Recommend plans & discounts for a specific user.
    Input JSON: { "user_id": "123" }
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    user_id = data.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    # Fetch ONLY this user's data
    user = users_collection.find_one({"user_id": user_id})
    if not user:
        return jsonify({"error": "User not found"}), 404

    current_plan = user.get("current_plan")
    previous_plans = user.get("history", [])
    duration = user.get("duration", 0)
    rating = user.get("rating", 0)
    engagement_time = user.get("engagement_time", 0)

    # Prepare features for ML model
    features = {
        "current_product_id": current_plan,
        "previous_products": len(previous_plans),
        "duration": duration,
        "rating": rating,
        "engagement_time": engagement_time
    }
    df = pd.DataFrame([features])

    # ML prediction → best discount
    prediction = model.predict(df.values)
    predicted_discount = prediction.tolist()

    # Fetch available discounts from DB
    discount_doc = discounts_collection.find_one({"plan": current_plan})
    available_discounts = discount_doc.get("discounts", []) if discount_doc else []

    return jsonify({
        "user_id": user_id,
        "current_plan": current_plan,
        "previous_plans": previous_plans,
        "predicted_best_discount": predicted_discount,
        "available_discounts": available_discounts
    })


# ======================
# ADMIN ENDPOINTS
# ======================
@app.route("/admin/most_preferred_product", methods=["GET"])
def most_preferred_product():
    """
    Admin: Find the most chosen current_plan among all users.
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    pipeline = [
        {"$group": {"_id": "$current_plan", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 1}
    ]
    result = list(users_collection.aggregate(pipeline))

    if not result:
        return jsonify({"message": "No data available"}), 404

    most_plan = result[0]["_id"]
    user_count = result[0]["count"]

    # Use ML model to validate trend for this plan
    df = pd.DataFrame([{
        "current_product_id": most_plan,
        "previous_products": 0,
        "duration": 5,
        "rating": 4,
        "engagement_time": 10
    }])
    prediction = model.predict(df.values)

    return jsonify({
        "most_preferred_product": most_plan,
        "user_count": user_count,
        "ai_suggestion": f"Predicted trend: {prediction.tolist()}"
    })


@app.route("/admin/plan_analytics", methods=["GET"])
def plan_analytics():
    """
    Admin: Return analytics for all plans (users count + avg duration + AI prediction).
    """
    pipeline = [
        {"$group": {
            "_id": "$current_plan",
            "users": {"$sum": 1},
            "avg_duration": {"$avg": "$duration"}
        }}
    ]
    result = list(users_collection.aggregate(pipeline))

    analytics = {}
    for item in result:
        plan = item["_id"]
        df = pd.DataFrame([{
            "current_product_id": plan,
            "previous_products": 0,
            "duration": item["avg_duration"],
            "rating": 4,
            "engagement_time": 10
        }])
        prediction = model.predict(df.values)
        analytics[plan] = {
            "users": item["users"],
            "avg_duration": item["avg_duration"],
            "ai_prediction": prediction.tolist()
        }

    return jsonify(analytics)


@app.route("/admin/optimize_discounts", methods=["GET"])
def optimize_discounts():
    """
    Admin: Use ML model to suggest discount optimization for all plans.
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    suggestions = {}
    all_discounts = discounts_collection.find()
    for discount in all_discounts:
        plan = discount["plan"]
        df = pd.DataFrame([{
            "current_product_id": plan,
            "previous_products": 0,
            "duration": 6,
            "rating": 4,
            "engagement_time": 12
        }])
        prediction = model.predict(df.values)
        suggestions[plan] = f"AI Suggestion → {prediction.tolist()}"

    return jsonify(suggestions)


# ======================
# GENERIC ML Prediction (Debug Only)
# ======================
@app.route("/predict", methods=["POST"])
def predict():
    """
    Raw prediction endpoint for testing with manual feature input.
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        prediction = model.predict(df.values)
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ======================
# Run Flask
# ======================
if __name__ == "__main__":
    app.run(debug=True, port=5000)
