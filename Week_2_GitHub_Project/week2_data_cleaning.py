from pathlib import Path
import pandas as pd

BASE_DIR=Path(__file__).resolve().parent
DATA_DIR=BASE_DIR/"data"; OUT=BASE_DIR/"output"
RAW=DATA_DIR/"raw_logistics_data.csv"; CLEAN=OUT/"clean_logistics_data.csv"; SUMMARY=OUT/"preprocessing_summary.txt"

def main():
    OUT.mkdir(exist_ok=True)
    df=pd.read_csv(RAW)
    raw_rows=len(df)
    missing_before=df.isna().sum()
    duplicate_count=int(df.duplicated().sum())

    df=df.drop_duplicates().copy()
    df["shipping_mode"]=df["shipping_mode"].astype("string").str.strip().str.title()

    numeric=["shipment_volume","distance_km","transport_cost","warehouse_load_pct","weather_risk","traffic_index","delivery_days"]
    for col in numeric:
        df[col]=df[col].fillna(df[col].median())

    q1=df["transport_cost"].quantile(.25); q3=df["transport_cost"].quantile(.75)
    iqr=q3-q1; lower=q1-1.5*iqr; upper=q3+1.5*iqr
    outliers=int(((df["transport_cost"]<lower)|(df["transport_cost"]>upper)).sum())
    df["transport_cost"]=df["transport_cost"].clip(lower,upper)

    df=pd.get_dummies(df,columns=["shipping_mode"],drop_first=False,dtype=int)
    df.to_csv(CLEAN,index=False)

    summary=[
        "WEEK 2 PREPROCESSING SUMMARY","="*30,
        f"Raw rows: {raw_rows}",f"Raw columns: {len(missing_before)}",
        f"Duplicate rows removed: {duplicate_count}","",
        "Missing values before cleaning:",missing_before.to_string(),"",
        "Missing values after cleaning:",df.isna().sum().to_string(),"",
        "Transport-cost IQR lower bound: %.2f"%lower,
        "Transport-cost IQR upper bound: %.2f"%upper,
        f"Transport-cost outliers capped: {outliers}","",
        f"Final dataset shape: {df.shape}"
    ]
    SUMMARY.write_text("\n".join(summary),encoding="utf-8")
    print("Cleaning completed successfully.")
    print(f"Rows: {len(df)} | Columns: {df.shape[1]}")
    print(f"Duplicates removed: {duplicate_count}")
    print(f"Outliers capped: {outliers}")
    print(f"Output: {CLEAN}")

if __name__=="__main__":
    main()
