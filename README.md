# Dollar Price Prediction using Machine Learning

This project predicts the future price of the US Dollar using Machine Learning models based on historical exchange rate data.

---

## 📌 Project Summary
The goal of this project is to analyze past USD exchange rate trends and predict future values using ML models.  
The project includes data preprocessing, visualization, model training, performance evaluation, and price forecasting.

---

## 🧠 Machine Learning Models Used
| Model              | Purpose |
|--------------------|---------|
| Linear Regression   | Baseline prediction model |
| Random Forest       | Improving prediction accuracy |

The model with highest performance is automatically selected and saved as `best_model.joblib`.

---

## 📊 Project Features
✔ Reads dollar price data from CSV  
✔ Trains multiple prediction models  
✔ Evaluates MAE & RMSE accuracy  
✔ Predicts future dollar price  
✔ Generates forecast visualization (`prediction_plot.png`)

---

## 🚀 How to Run
```bash
pip install -r requirements.txt
python predict_dollar.py

📂 File Structure
├── data.csv
├── predict_dollar.py
├── best_model.joblib
├── prediction_plot.png
└── README.md
📈 Results
After training the models, the system:

Selects the most accurate model
Saves it for future predictions
Displays forecasted price for the next day
A sample output:

Predicted dollar price for the next day:  xxxxx
🔮 Next Improvements (Future Work)
Add more ML models (LSTM Neural Network)
Add dashboard for live predictions
Deploy the model on a website or API
Integrate crypto price forecasting

👤 Developer
Mahdis Tirgari
AI & Machine Learning Developer

⭐ Support
If you liked this project, don’t forget to give the repository a ⭐ on GitHub!