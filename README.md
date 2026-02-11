# 🚗 Used Car Price Predictor

A machine learning web application that predicts the resale price of used cars using a trained **XGBoost regression model**, deployed via a clean **Streamlit** interface.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-orange.svg)

---

## 🔍 Overview

This project takes real-world used car features as input — brand, fuel type, mileage, engine specs, accident history, and more — and outputs an estimated resale price. The model was trained on a dataset of used car listings and uses log-transformed price prediction for improved accuracy.

---

## 🧠 ML Pipeline

- **Algorithm:** XGBoost Regressor
- **Target Variable:** `log(price)` → converted back with `expm1`
- **Feature Engineering:**
  - Brand target encoding (`brand_encoded`)
  - Vehicle age bucketing (`Age_Mid`, `Age_Old`, `Age_Very_Old`)
  - Mileage tier encoding (`Milage_Medium`, `Milage_High`, `Milage_Very_High`)
  - Accident impact scoring (`Accident_Impact`)
  - Engine type flag (`is_v_engine`)
- **Artifacts:** Model, column order, scaler, and brand mapping saved via `joblib`

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| ML Model | XGBoost |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Web App | Streamlit |
| Serialization | Joblib |
| Notebook | Jupyter |

---

## 📁 Project Structure

```
used-car-price-predictor/
│
├── app.py                         # Streamlit web app
├── used_car_prediction.ipynb      # Training notebook
├── used_cars.csv                  # Raw dataset
│
├── used_car_price_xgboost.pkl     # Trained XGBoost model
├── column.pkl                     # Feature column order
├── brand_mapping.pkl              # Brand target encoding map
└── scaler.pkl                     # Feature scaler
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Namanbansal9414/used-car-price-predictor.git
cd used-car-price-predictor
```

### 2. Install dependencies

```bash
pip install streamlit pandas numpy xgboost scikit-learn joblib
```

### 3. Run the app

```bash
streamlit run app.py
```

---

## 🖥️ App Features

- Select car **brand**, **fuel type**, **transmission**, and **colors**
- Enter **horsepower**, **engine displacement**, and **vehicle age**
- Specify **mileage per year** and **accident history**
- Click **Predict Price** to get an estimated resale value in ₹

---

## 📊 Input Features

| Feature | Type | Description |
|---|---|---|
| Brand | Categorical | Car manufacturer |
| Fuel Type | Categorical | Petrol / Diesel / Electric / Hybrid / CNG |
| Transmission | Categorical | Manual / Automatic |
| Exterior Color | Categorical | Body color |
| Interior Color | Categorical | Cabin color |
| Clean Title | Binary | Whether title is clean |
| Horsepower | Numeric | Engine power (HP) |
| Engine Displacement | Numeric | Engine size (Liters) |
| V Engine | Binary | Whether engine is V-type |
| Accident History | Ordinal | None / Minor / Major |
| Vehicle Age | Numeric | Years since manufacture |
| Mileage Per Year | Numeric | Average km driven per year |

---

## 📌 Notes

- Price is predicted in **USD** internally and displayed in **INR (₹)** using a fixed conversion rate (×90).
- Brand encoding uses **target mean encoding** — unseen brands fall back to the global mean.
- All encodings in `app.py` exactly mirror the training pipeline to avoid data leakage.

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
---

## 👤 Author

Your Name
- GitHub: [@NamanBansal9414](https://github.com/Namanbansal9414?tab=repositories)
- LinkedIn: [@NamanBansal9509](https://www.linkedin.com/in/namanbansal9509/)


---
## 📧 Contact

For questions or feedback, please open an issue on GitHub and direct DM on LinkedIn.


---