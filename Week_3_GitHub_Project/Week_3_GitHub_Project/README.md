# Logistics Data Analyst Internship — Week 3
## Advanced Data Analysis and Visualization

Technical project for Week 3.

### Objective
Analyze logistics performance using descriptive statistics, KPI calculations, relationship analysis and visualizations.

### Examples
1. **KPI analysis:** shipment volume, on-time delivery rate, average delivery time, average transport cost and average distance.
2. **Shipping-mode comparison:** mean, median and standard deviation of delivery time by shipping mode.
3. **Relationship analysis:** correlation between distance and delivery time plus a scatter plot.

### Visualizations
- Delivery-time distribution
- Average delivery time by shipping mode
- Distance versus delivery time

### Dataset
`data/logistics_analysis_data.csv` contains 1,000 synthetic shipment records for reproducible technical demonstration. It is not proprietary company data.

### Run
```bash
pip install -r requirements.txt
python week3_analysis.py
```

The script creates:
```text
output/
├── 01_delivery_time_distribution.png
├── 02_delivery_by_shipping_mode.png
├── 03_distance_vs_delivery.png
├── kpi_summary.csv
├── shipping_mode_summary.csv
└── correlation_matrix.csv
```

### Business questions
- What is the typical delivery time?
- What percentage meets the service target?
- Which shipping mode is fastest?
- Is distance associated with delivery time?
- Which variables should be investigated in predictive modeling?

### Note
All numerical results are examples based on synthetic data.

### Structure
```text
Week_3_GitHub_Project/
├── README.md
├── week3_analysis.py
├── requirements.txt
├── .gitignore
├── data/
├── output/
└── report/
```
