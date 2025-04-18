
# 🛡️ DisasterPrep AI

**DisasterPrep AI** is a machine learning-powered Streamlit application that helps individuals assess their household's preparedness for natural disasters. It uses custom checklists, state-specific disaster risks, and user inputs to provide personalized preparedness levels and actionable recommendations.

## 🚀 Features

- ✅ Predicts preparedness level: *Well Prepared*, *Moderately Prepared*, or *Needs Urgent Prep*  
- 📍 Selects disaster risk tier automatically based on Indian state and disaster type  
- 📋 Interactive checklist with personalized completion tracking  
- 💡 Displays improvement tips and recommendations for each disaster type  
- 🧠 Built with XGBoost classifier trained on synthetic disaster readiness data

## 🧰 Tech Stack

- **Python 3.10+**
- **Streamlit**
- **XGBoost**
- **scikit-learn**
- **pandas**
- **joblib**

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/your-username/disasterprep-ai.git
cd disasterprep-ai

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📁 Project Structure

```
disasterprep-ai/
├── app.py                  # Streamlit frontend
├── model/
│   ├── disaster_preparedness_xgb_model.pkl
│   ├── label_encoder.pkl
│   ├── model_features.pkl
│   └── checklists.pkl
├── requirements.txt
└── README.md
```
## 🧠 How It Works

1. User selects their **state** and **disaster type**
2. The app auto-detects the **risk tier** for that region-disaster combo
3. User fills out a checklist of preparedness actions
4. ML model predicts preparedness level based on inputs
5. Personalized tips and actions are displayed to improve readiness

## 🤝 Contributing

Got ideas to improve this tool?  
Pull requests are welcome! Please open an issue first to discuss what you would like to change.

## 📜 License

This project is licensed under the MIT License.
