# Logistics Data Analyst Internship — Week 4
## Predictive Modeling and Optimization

Final technical project demonstrating delivery-time regression, on-time-delivery classification, model evaluation, example prediction, and a simple shipping-mode scenario analysis.

### Models
- Random Forest Regression → delivery time
- Random Forest Classification → on-time status
- 80/20 train-test split (`random_state=42`)
- One-hot encoding for shipping mode

### Features
`shipment_volume`, `distance_km`, `warehouse_load_pct`, `weather_risk`, `traffic_index`, `shipping_mode`

### Evaluation
Regression: MAE, RMSE, R²  
Classification: Accuracy, Precision, Recall, F1 and confusion matrix.

### Example
The script predicts a sample shipment (180 volume, 650 km, 72% warehouse load, weather risk 1, traffic 55, First Class), then compares all four shipping modes under the same conditions.

### Run
```bash
pip install -r requirements.txt
python week4_predictive_modeling.py
```

Outputs: model metrics, test predictions, example prediction, shipping-mode scenario analysis, confusion matrix and actual-vs-predicted chart.

### Data note
The dataset is synthetic and used for reproducible educational/technical demonstration. Numerical results do not represent actual company performance. The scenario comparison is a simple optimization example, not a full mathematical optimizer.

### Structure
```text
Week_4_GitHub_Project/
├── README.md
├── week4_predictive_modeling.py
├── requirements.txt
├── .gitignore
├── data/
├── output/
└── report/
```
