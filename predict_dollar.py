# predict_dollar.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import matplotlib.pyplot as plt
df = pd.read_csv('dollar.csv.txt', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)
for lag in range(1, 6):
    df[f'lag_{lag}'] = df['price'].shift(lag)
df = df.dropna().reset_index(drop=True)
features = [f'lag_{lag}' for lag in range(1, 6)]
X = df[features]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
def evaluate(true, pred, name):
    mae = mean_absolute_error(true, pred)
    rmse = mean_squared_error(true, pred)
    print(f'== {name} ==') 
    print('MAE:', round(mae, 2))
    print('RMSE:', round(rmse, 2))
    print()
evaluate(y_test, y_pred_lr, 'LinearRegression')
evaluate(y_test, y_pred_rf, 'RandomForest')
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
best_model = rf if mae_rf < mae_lr else lr
best_name = 'RandomForest' if mae_rf < mae_lr else 'LinearRegression'
print('Best model:', best_name)
joblib.dump(best_model, 'best_model.joblib')
print('Saved best model to best_model.joblib')
plt.figure(figsize=(8,4))
plt.plot(df['date'].iloc[-len(y_test):], y_test.values, label='Actual')
plt.plot(df['date'].iloc[-len(y_test):], y_pred_rf, label='Predicted (RF)')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('prediction_plot.png')
print('Saved prediction plot to prediction_plot.png')
last_row = df.iloc[-1]
input_for_next = np.array([last_row[f'lag_{i}'] for i in range(1,6)]).reshape(1,-1)
pred_next = best_model.predict(input_for_next)
print('Predicted next price:', round(pred_next[0], 2))
