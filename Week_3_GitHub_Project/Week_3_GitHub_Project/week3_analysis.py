from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR=Path(__file__).resolve().parent
DATA_FILE=BASE_DIR/"data/logistics_analysis_data.csv"
OUTPUT_DIR=BASE_DIR/"output"

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df=pd.read_csv(DATA_FILE)

    print("=== DATASET OVERVIEW ===")
    print("Rows:",len(df))
    print("Columns:",df.shape[1])
    print("\nMissing values:")
    print(df.isna().sum())

    kpis=pd.DataFrame({
        "Metric":["Shipment Volume","On-Time Delivery Rate (%)","Average Delivery Time (days)","Average Transport Cost","Average Distance (km)"],
        "Value":[len(df),df["on_time"].mean()*100,df["delivery_days"].mean(),df["transport_cost"].mean(),df["distance_km"].mean()]
    })
    print("\n=== KPI EXAMPLES ===")
    print(kpis.round(2).to_string(index=False))
    kpis.to_csv(OUTPUT_DIR/"kpi_summary.csv",index=False)

    mode_summary=df.groupby("shipping_mode")["delivery_days"].agg(["count","mean","median","std"]).round(2).reset_index()
    print("\n=== SHIPPING MODE SUMMARY ===")
    print(mode_summary.to_string(index=False))
    mode_summary.to_csv(OUTPUT_DIR/"shipping_mode_summary.csv",index=False)

    corr_cols=["shipment_volume","distance_km","transport_cost","warehouse_load_pct","weather_risk","traffic_index","delivery_days"]
    df[corr_cols].corr().round(3).to_csv(OUTPUT_DIR/"correlation_matrix.csv")

    plt.figure(figsize=(8,5))
    plt.hist(df["delivery_days"],bins=25)
    plt.xlabel("Delivery Days"); plt.ylabel("Number of Shipments")
    plt.title("Delivery-Time Distribution"); plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"01_delivery_time_distribution.png",dpi=150); plt.close()

    mode_avg=df.groupby("shipping_mode")["delivery_days"].mean().sort_values()
    plt.figure(figsize=(8,5))
    mode_avg.plot(kind="bar")
    plt.xlabel("Shipping Mode"); plt.ylabel("Average Delivery Days")
    plt.title("Average Delivery Time by Shipping Mode")
    plt.xticks(rotation=30,ha="right"); plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"02_delivery_by_shipping_mode.png",dpi=150); plt.close()

    plt.figure(figsize=(8,5))
    plt.scatter(df["distance_km"],df["delivery_days"],alpha=.35,s=18)
    plt.xlabel("Distance (km)"); plt.ylabel("Delivery Days")
    plt.title("Distance vs Delivery Time"); plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"03_distance_vs_delivery.png",dpi=150); plt.close()

    distance_corr=df["distance_km"].corr(df["delivery_days"])
    print("\n=== EXAMPLE INSIGHTS ===")
    print(f"Average delivery time: {df['delivery_days'].mean():.2f} days")
    print(f"On-time rate at 5-day threshold: {df['on_time'].mean()*100:.2f}%")
    print(f"Distance/delivery-time correlation: {distance_corr:.3f}")
    print("\nCharts and summaries saved in output/")

if __name__=="__main__":
    main()
