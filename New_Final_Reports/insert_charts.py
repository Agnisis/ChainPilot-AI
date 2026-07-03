import re

file_path = "c:\\Users\\agnis\\OneDrive\\Desktop\\My Workspace\\SCMAi\\New_Final_Reports\\05_Chapter5_Results_Discussion.md"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Remove old placeholders like > **[INSERT Figure...
content = re.sub(r'> \*\*\[INSERT Figure.*?\n.*?\n.*?\n', '', content)

# 1. EDA Insertions
eda_images = """
![Demand Trend Over Time](Charts_Graphs/Demand_Trend.png)
![Time Series Trend Decomposition](Charts_Graphs/Time_Series_Trend_Decomposition.png)
![Rolling Mean Analysis](Charts_Graphs/Rolling_Mean.png)
![Rolling Standard Deviation Analysis](Charts_Graphs/Rolling_Standard_Deviation.png)
"""
content = content.replace("#### 5.1.2 Demand Trend Analysis\n", "#### 5.1.2 Demand Trend Analysis\n" + eda_images)

corr_image = "\n![Feature Correlation Heatmap](Charts_Graphs/Coorelation_Heatmap.png)\n"
content = content.replace("#### 5.1.3 Feature Correlation Analysis\n", "#### 5.1.3 Feature Correlation Analysis\n" + corr_image)

dist_images = """
![Top Products By Sale](Charts_Graphs/top_Products_By_Sale.png)
![Profit By Region](Charts_Graphs/profit_By_region.png)
"""
content = content.replace("#### 5.1.4 Data Distribution Analysis\n", "#### 5.1.4 Data Distribution Analysis\n" + dist_images)

# 2. Metrics Insertions
metrics_images = """
![RMSE Comparison Across Models](Charts_Graphs/RMSE_Comparison.png)
![MAE Comparison Across Models](Charts_Graphs/MAE_Comparison.png)
![MAPE Comparison Across Models](Charts_Graphs/MAPE_Comparison.png)
![R2 Comparison Across Models](Charts_Graphs/R2_Compariosn.png)
"""
content = content.replace("#### 5.2.1 RMSE Comparison Across Models\n", "#### 5.2.1 RMSE Comparison Across Models\n" + metrics_images)

# 3. Actual vs Predicted Insertions
actual_pred_images = """
![ARIMA Actual vs Predicted](Charts_Graphs/ARIMA_Actual_vs_Predicted.png)
![SARIMA Actual vs Predicted](Charts_Graphs/SARIMA_Actual_vs_Predicted.png)
![Random Forest Actual vs Predicted](Charts_Graphs/Random_Forest_Actual_vs_Predicted.png)
![LightGBM Actual vs Predicted](Charts_Graphs/LightGBM_Actual_vs_Predicted.png)
![XGBoost Actual vs Predicted](Charts_Graphs/XGBoost_Actual_vs_Predicted.png)
"""
content = content.replace("#### 5.3.1 Actual vs Predicted Demand\n", "#### 5.3.1 Actual vs Predicted Demand\n" + actual_pred_images)

# 4. Anomaly Detection
anomaly_image = "\n![Isolation Forest Anomaly Detection](Charts_Graphs/Demand_Anomaly_Detection.png)\n"
content = content.replace("#### 5.4.1 Multivariate Anomaly Scatter\n", "#### 5.4.1 Multivariate Anomaly Scatter\n" + anomaly_image)

# 5. Future Forecasting
forecast_section = """
#### 5.5.2 90-Day Future Forecast
The following chart demonstrates the system's capability to project historical demand 90 days into the future using the best performing model (Optuna-tuned XGBoost).

![Historical Demand vs 90-Day Forecast](Charts_Graphs/Historial_Demand_vs_90_Days_Forcast.png)
"""
content = content.replace("### 5.5 Web Application Dashboard\n", "### 5.5 Web Application Dashboard\n" + forecast_section)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Successfully injected 18 charts into Chapter 5.")
