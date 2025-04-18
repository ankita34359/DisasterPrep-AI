from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and encoders
model = joblib.load('disaster_preparedness_xgb_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
model_features = joblib.load('model_features.pkl')

# Define disaster risk map and custom checklist
disaster_risk_map = {
    "Assam": ["Flood", "Earthquake", "Landslide"],
    "Uttarakhand": ["Landslide", "Flood", "Earthquake"],
    "Gujarat": ["Earthquake", "Drought", "Severe Storm"],
    "Tamil Nadu": ["Cyclone", "Flood", "Heatwave"],
    "West Bengal": ["Cyclone", "Flood", "Landslide"],
    "Delhi": ["Heatwave", "Cold Wave"],
    "Rajasthan": ["Drought", "Heatwave"],
    "Bihar": ["Flood", "Heatwave", "Cold Wave"],
    "Maharashtra": ["Drought", "Cyclone", "Severe Storm"],
    "Andhra Pradesh": ["Cyclone", "Flood", "Severe Storm"],
    "Kerala": ["Landslide", "Flood"],
    "Himachal Pradesh": ["Landslide", "Cold Wave"],
    "Odisha": ["Cyclone", "Flood"],
    "Telangana": ["Heatwave", "Drought"],
    "Jharkhand": ["Drought", "Cold Wave"],
    "Punjab": ["Cold Wave", "Flood"],
    "Goa": ["Cyclone", "Flood"]
}

risk_tiers = {
    "High": ["Earthquake", "Cyclone", "Flood"],
    "Medium": ["Landslide", "Severe Storm", "Industrial Hazard"],
    "Low": ["Drought", "Heatwave", "Cold Wave"]
}

custom_checklists = {
    "Earthquake": [
        "Secured heavy furniture", "Learned how to turn off utilities", "Emergency contact numbers known",
        "Prepared evacuation plan", "Kept shoes and flashlight beside bed", "First aid kit ready",
        "Important documents accessible", "Practiced earthquake drill", "Know nearest safe zone",
        "Know region's seismic history"
    ],
    "Flood": [
        "Know flood evacuation routes", "Have waterproof bags for documents", "First aid kit ready",
        "Food & water supplies", "Follow flood alerts", "Elevated electrical appliances",
        "Flood insurance taken", "Practiced flood drill", "Emergency contact numbers known",
        "Nearby shelters identified"
    ],
    "Cyclone": [
        "Reinforced windows and doors", "Tree branches trimmed", "Evacuation kit ready",
        "Important documents secured", "Battery-operated radio available", "Emergency contact numbers known",
        "Cyclone alerts followed", "Family trained on safety steps", "Mock drill done", "Water storage prepared"
    ],
    "Drought": [
        "Rainwater harvesting in place", "Water usage optimized", "Stored water for drinking",
        "Drip irrigation used", "Drought-resilient crops selected", "Emergency water plan",
        "Family informed of drought coping methods", "Local drought alerts followed",
        "Check wells and pumps", "Stored food"
    ],
    "Cold Wave": [
        "Warm clothing ready", "Heaters in safe condition", "Insulated home",
        "Emergency contact numbers known", "Backup heating source", "Food & water supplies",
        "Family trained for cold exposure", "Medical needs addressed", "Pets prepared", "Followed cold alerts"
    ],
    "Heatwave": [
        "Hydration plan followed", "Access to cool areas", "Avoided outdoor work during peak hours",
        "First aid for heatstroke known", "Family educated on symptoms", "Fans and AC functional",
        "Windows shaded", "Light cotton clothes used", "Followed heatwave alerts", "Mock drill conducted"
    ],
    "Landslide": [
        "Monitored slope signs", "Retaining walls checked", "Evacuation plan ready",
        "Emergency contact numbers known", "Drainage paths cleared", "Important items secured",
        "Followed weather updates", "Nearby shelters identified", "Practiced landslide drill",
        "Avoided unstable ground"
    ],
    "Severe Storm": [
        "Trimmed trees and shrubs", "Secured outdoor objects", "Emergency kit ready",
        "Listened to storm warnings", "Safe room identified", "Power backups ready",
        "Important papers waterproofed", "Mock drill done", "First aid kit ready", "Emergency contacts updated"
    ],
    "Industrial Hazard": [
        "Know local industry risks", "Toxic leak evacuation plan", "Gas masks and filters ready",
        "Government alerts followed", "Emergency contact numbers known", "Safe routes identified",
        "Community drill participated", "Important documents safe", "Family trained", "Nearby hospitals listed"
    ]
}

def calculate_completion(checklist_responses):
    return sum(checklist_responses.values()) / len(checklist_responses) * 100

def get_improvement_tips(level):
    tips = {
        "Needs Urgent Prep": [
            "Complete all urgent checklist items immediately.",
            "Create an evacuation plan and share with family."
        ],
        "Moderately Prepared": [
            "Review missing checklist items.",
            "Conduct a disaster drill with household members."
        ],
        "Well Prepared": [
            "Stay updated with local disaster alerts.",
            "Refresh emergency supplies every 6 months."
        ]
    }
    return tips.get(level, [])

def get_recommendations(disaster_type, level):
    checklist = custom_checklists.get(disaster_type, [])
    if level == "Needs Urgent Prep":
        return checklist[:3]
    elif level == "Moderately Prepared":
        return checklist[3:6]
    return checklist

def get_risk_tier(disaster_type):
    for tier, disasters in risk_tiers.items():
        if disaster_type in disasters:
            return {"High": 2, "Medium": 1, "Low": 0}[tier]
    return 1  # Default to Medium if unknown

@app.route('/api/states', methods=['GET'])
def get_states():
    """Return list of available states"""
    return jsonify({
        "states": list(disaster_risk_map.keys())
    })

@app.route('/api/disasters', methods=['GET'])
def get_disasters():
    """Return list of disasters for a given state"""
    state = request.args.get('state')
    if not state or state not in disaster_risk_map:
        return jsonify({"error": "Invalid state"}), 400
    
    return jsonify({
        "disasters": disaster_risk_map[state]
    })

@app.route('/api/checklist', methods=['GET'])
def get_checklist():
    """Return checklist items for a given disaster type"""
    disaster_type = request.args.get('disaster_type')
    if not disaster_type or disaster_type not in custom_checklists:
        return jsonify({"error": "Invalid disaster type"}), 400
    
    return jsonify({
        "checklist": custom_checklists[disaster_type]
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make preparedness prediction based on user input"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['state', 'disaster_type', 'household_size', 'has_kit', 'checklist_responses']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Calculate derived values
        risk_tier = get_risk_tier(data['disaster_type'])
        awareness_score = sum(data['checklist_responses'].values())
        completion = calculate_completion(data['checklist_responses'])

        # Prepare features for model
        features_df = pd.DataFrame([{
            "Household Size": data['household_size'],
            "Disaster Kit Owned": int(data['has_kit']),
            "Checklist Completion %": completion,
            "Awareness Score": awareness_score,
            "Risk Tier": risk_tier
        }])

        # Make prediction
        prediction = model.predict(features_df)[0]
        level = label_encoder.inverse_transform([prediction])[0]

        # Prepare response
        response = {
            "preparedness_level": level,
            "improvement_tips": get_improvement_tips(level),
            "recommendations": get_recommendations(data['disaster_type'], level),
            "completion_percentage": completion,
            "awareness_score": awareness_score
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)