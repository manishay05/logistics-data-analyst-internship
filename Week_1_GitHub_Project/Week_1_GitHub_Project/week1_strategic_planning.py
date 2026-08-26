"""
Week 1 - Strategic Planning and Data Exploration
Logistics Data Analyst Internship

This script loads the synthetic Week 1 sample dataset and produces
basic KPI calculations and exploratory summaries.
"""

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "logistics_sample_data.csv"


def load_data():
    """Load the Week 1 sample logistics dataset."""
    return pd.read_csv(DATA_FILE)


def calculate_kpis(df):
    """Calculate the KPIs defined in the Week 1 project plan."""
    return {
        "shipment_volume": len(df),
        "on_time_delivery_rate_pct": df["on_time"].mean() * 100,
        "average_delivery_time_days": df["delivery_days"].mean(),
        "transport_cost_per_shipment": df["transport_cost"].mean(),
        "late_delivery_rate_pct": (1 - df["on_time"]).mean() * 100,
    }


def main():
    df = load_data()

    print("\n=== DATASET OVERVIEW ===")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print("\nColumns:")
    print(list(df.columns))

    print("\n=== DESCRIPTIVE STATISTICS ===")
    print(df.describe(numeric_only=True).round(2))

    print("\n=== SHIPPING MODE COUNTS ===")
    print(df["shipping_mode"].value_counts())

    print("\n=== WEEK 1 KPI SUMMARY ===")
    kpis = calculate_kpis(df)
    for name, value in kpis.items():
        print(f"{name}: {value:.2f}" if isinstance(value, float) else f"{name}: {value}")

    print("\n=== STRATEGIC QUESTIONS FOR NEXT WEEKS ===")
    print("1. Which factors are associated with longer delivery times?")
    print("2. Which shipping modes provide the best service/cost trade-off?")
    print("3. Do distance, traffic and warehouse load predict delivery time?")
    print("4. Which shipments should be prioritized for intervention?")


if __name__ == "__main__":
    main()
