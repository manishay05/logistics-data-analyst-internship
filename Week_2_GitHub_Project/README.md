# Logistics Data Analyst Internship — Week 2
## Data Collection, Cleaning and Preprocessing

This is the technical GitHub project for Week 2.

### Objective
Prepare reliable logistics data for analysis and machine learning by identifying and treating common data-quality problems.

### What is demonstrated
- Data-quality inspection
- Missing-value detection and median imputation
- Duplicate detection and removal
- Categorical standardization
- IQR-based outlier detection and capping
- One-hot encoding
- Reproducible output generation
- Preprocessing documentation

### Dataset
`data/raw_logistics_data.csv` is a synthetic dataset intentionally created with missing values, inconsistent shipping-mode labels, duplicate records and transport-cost outliers. It is used only for technical demonstration and is not presented as real company data.

### Run
```bash
pip install -r requirements.txt
python week2_data_cleaning.py
```

The script generates:
```text
output/clean_logistics_data.csv
output/preprocessing_summary.txt
```

### Workflow
1. Load raw data.
2. Inspect missing values and duplicates.
3. Remove exact duplicates.
4. Standardize shipping-mode labels.
5. Impute missing numeric values with medians.
6. Detect and cap transport-cost outliers using IQR.
7. One-hot encode shipping mode.
8. Save cleaned data and a preprocessing summary.

### Production note
In a production ML pipeline, preprocessing statistics should be learned from training data only. Target leakage must also be avoided by excluding variables that reveal the outcome after delivery.

### Repository structure
```text
Week_2_GitHub_Project/
├── README.md
├── week2_data_cleaning.py
├── requirements.txt
├── .gitignore
├── data/
│   └── raw_logistics_data.csv
├── output/
│   └── generated after running the script
└── report/
    └── Week_2_Data_Collection_Cleaning_and_Preprocessing.docx
```
