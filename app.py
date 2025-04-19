import streamlit as st
import joblib
import pandas as pd

# Load model and encoders
model = joblib.load('disaster_preparedness_xgb_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
model_features = joblib.load('model_features.pkl')

# Define disaster risk map and custom checklist
all_disasters = [
    "Flood", "Earthquake", "Landslide", "Drought", "Severe Storm", 
    "Cyclone", "Heatwave", "Cold Wave", "Industrial Hazard"
]

states_and_uts = [
    # States
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",

    # Union Territories
    "Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu", 
    "Delhi", "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry"
]

disaster_risk_map = {region: all_disasters for region in states_and_uts}




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

custom_recommendations = {
    "Earthquake": [
        "Secure your home structure with a professional inspection.",
        "Install seismic shut-off valves for gas lines.",
        "Join a local community earthquake awareness group."
    ],
    "Flood": [
        "Install water sensors in flood-prone areas.",
        "Keep sandbags ready for quick use.",
        "Subscribe to SMS flood alerts from local authorities."
    ],
    "Cyclone": [
        "Reinforce your home's roofing materials.",
        "Establish a cyclone-safe room in your home.",
        "Register for community cyclone warning updates."
    ],
    "Drought": [
        "Invest in a water-efficient irrigation system.",
        "Monitor local groundwater levels regularly.",
        "Collaborate with neighbors to manage water usage."
    ],
    "Cold Wave": [
        "Insulate plumbing to prevent freezing.",
        "Check on elderly neighbors during extreme cold.",
        "Register for cold wave alerts from the local government."
    ],
    "Heatwave": [
        "Create a cool room with blackout curtains and fans.",
        "Coordinate with neighbors for mutual aid during heat spikes.",
        "Join a local heatwave preparedness campaign."
    ],
    "Landslide": [
        "Install proper drainage systems around your home.",
        "Avoid heavy construction near slopes.",
        "Use vegetation to help stabilize soil."
    ],
    "Severe Storm": [
        "Anchor outdoor structures like sheds or swings.",
        "Get your roof inspected before storm season.",
        "Keep mobile phone power banks charged at all times."
    ],
    "Industrial Hazard": [
        "Attend safety training from local industries.",
        "Install indoor air quality sensors.",
        "Keep an emergency go-bag with safety gear and essentials."
    ]
}

def get_recommendations(disaster_type, level):
    return custom_recommendations.get(disaster_type, [])


def get_risk_tier(disaster_type):
    for tier, disasters in risk_tiers.items():
        if disaster_type in disasters:
            return {"High": 2, "Medium": 1, "Low": 0}[tier]
    return 1  

# --- Streamlit UI ---
st.title("🛡️ Disaster Preparedness Level Predictor")

st.markdown("Enter your household information to assess your preparedness level.")

state = st.selectbox("Select your State", list(disaster_risk_map.keys()))
disaster_type = st.selectbox("Select Disaster Type", disaster_risk_map[state])

household_size = st.slider("Household Size", 1, 15, 4)
has_kit = st.radio("Do you have a disaster kit?", ["Yes", "No"]) == "Yes"

checklist_items = custom_checklists.get(disaster_type, [])
checklist_responses = {}
st.markdown("### Checklist - Mark items you've completed:")
for item in checklist_items:
    checklist_responses[item] = st.checkbox(item)

if st.button("🔍 Predict Preparedness"):
    risk_tier = get_risk_tier(disaster_type)
    awareness_score = sum(checklist_responses.values())
    completion = calculate_completion(checklist_responses)

    features_df = pd.DataFrame([{
        "Household Size": household_size,
        "Disaster Kit Owned": int(has_kit),
        "Checklist Completion %": completion,
        "Awareness Score": awareness_score,
        "Risk Tier": risk_tier
    }])

    prediction = model.predict(features_df)[0]
    level = label_encoder.inverse_transform([prediction])[0]

    st.success(f"### 🎯 You are: **{level}**")

    st.markdown("#### 📌 Improvement Tips:")
    for tip in get_improvement_tips(level):
        st.write("- " + tip)

    st.markdown("#### ✅ Recommended Actions:")
    for action in get_recommendations(disaster_type, level):
        st.write("🔹", action)



