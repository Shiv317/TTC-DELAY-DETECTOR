# 🚇 TTC Subway Delay Risk App

**Predict subway delays. Visualize risk. Understand causes.**  
A smart ML-powered dashboard to help Toronto commuters and city planners assess subway delay risks by line, hour, and day — with explainable AI at its core.

![image](https://github.com/user-attachments/assets/1c700e0e-2f29-4fa9-a69e-c594d8bac46f)
![image](https://github.com/user-attachments/assets/168b625b-424d-411a-9528-bedf523da595)



---

## 📌 Features

- 🔮 **Predict delay probability** for TTC subway lines based on:
  - Line (`YU`, `BD`, `SHP`, `SRT`)
  - Time of day
  - Day of the week
- 📍 **Live Mapbox visualization** to show spatial delay risk
- 🧠 **SHAP explanations** to interpret what drives each prediction
- 💬 **Natural language summary** of the top contributing factor
- 📊 **Top 5 most common delay causes** chart from historical TTC data

---

## 🛠️ Tech Stack

| Component         | Tools/Libraries                     |
|------------------|--------------------------------------|
| Machine Learning | `XGBoost`, `scikit-learn`, `SHAP`    |
| Frontend UI      | `Streamlit`, `PyDeck`, `Plotly`      |
| Map Integration  | `Mapbox`, `PyDeck`                   |
| Dataset Source   | [Toronto Open Data – TTC Delays](https://open.toronto.ca/dataset/ttc-subway-delay-data/) |

---

## 📂 File Structure

ttc-delay-app/
├── app.py # Streamlit app
├── model.pkl # Trained XGBoost model
├── ttc_final_dataset.csv # Cleaned & feature-engineered TTC data
├── requirements.txt # All dependencies
└── README.md


## 🚀 How to Run

1. git clone [https://github.com/shiv317/ttc-delay-app.git](https://github.com/Shiv317/TTC-DELAY-DETECTOR)
cd ttc-delay-app

2. python -m venv venv
venv\Scripts\activate

3. pip install -r requirements.txt

4. streamlit run app.py


Why I Built This ?
Toronto commuters lose hours to transit delays — but there's no predictive insight on when/where they're likely. This project uses AI to:

Forecast risk in real time

Help citizens make better decisions

Show how interpretable ML can improve public services


Author
Shivail Anand
BSc Computer Science, York University
Aspiring ML Engineer | AI Enthusiast

If you found this project helpful...
Please consider giving it a ⭐ on GitHub — it really helps!




