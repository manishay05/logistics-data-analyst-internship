from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE=Path(__file__).resolve().parent; DATA=BASE/"data/logistics_modeling_data.csv"; OUT=BASE/"output"
FEATURES=["shipment_volume","distance_km","warehouse_load_pct","weather_risk","traffic_index","shipping_mode"]

def prep():
    return ColumnTransformer([("num", "passthrough", FEATURES[:-1]),("cat",OneHotEncoder(handle_unknown="ignore"),["shipping_mode"])])

def main():
    OUT.mkdir(exist_ok=True); df=pd.read_csv(DATA)
    X=df[FEATURES]; yr=df["delivery_days"]; yc=df["on_time"]
    Xtr,Xte,yrtr,yrte,yctr,ycte=train_test_split(X,yr,yc,test_size=.2,random_state=42)
    reg=Pipeline([("prep",prep()),("model",RandomForestRegressor(n_estimators=150,max_depth=10,random_state=42))])
    clf=Pipeline([("prep",prep()),("model",RandomForestClassifier(n_estimators=150,max_depth=10,random_state=42))])
    reg.fit(Xtr,yrtr); clf.fit(Xtr,yctr)
    pr=reg.predict(Xte); pc=clf.predict(Xte)
    rm={"MAE_days":mean_absolute_error(yrte,pr),"RMSE_days":mean_squared_error(yrte,pr)**.5,"R2":r2_score(yrte,pr)}
    cm={"Accuracy":accuracy_score(ycte,pc),"Precision":precision_score(ycte,pc,zero_division=0),"Recall":recall_score(ycte,pc,zero_division=0),"F1":f1_score(ycte,pc,zero_division=0)}
    print("=== REGRESSION ==="); [print(f"{k}: {v:.4f}") for k,v in rm.items()]
    print("=== CLASSIFICATION ==="); [print(f"{k}: {v:.4f}") for k,v in cm.items()]
    pd.DataFrame([{"model":"Delivery Time Regression","metric":k,"value":v} for k,v in rm.items()]+[{"model":"On-Time Classification","metric":k,"value":v} for k,v in cm.items()]).to_csv(OUT/"model_metrics.csv",index=False)
    pred=Xte.reset_index(drop=True).copy(); pred["actual_delivery_days"]=yrte.reset_index(drop=True); pred["predicted_delivery_days"]=pr; pred["actual_on_time"]=ycte.reset_index(drop=True); pred["predicted_on_time"]=pc; pred.to_csv(OUT/"test_predictions.csv",index=False)
    example=pd.DataFrame([{"shipment_volume":180,"distance_km":650,"warehouse_load_pct":72,"weather_risk":1,"traffic_index":55,"shipping_mode":"First Class"}])
    example_out=example.copy(); example_out["predicted_delivery_days"]=reg.predict(example); example_out["predicted_on_time"]=clf.predict(example); example_out.to_csv(OUT/"example_shipment_prediction.csv",index=False)
    scenarios=[]
    for mode in ["Same Day","First Class","Second Class","Standard"]:
        s=example.copy(); s["shipping_mode"]=mode; s["predicted_delivery_days"]=reg.predict(s); scenarios.append(s)
    pd.concat(scenarios,ignore_index=True).sort_values("predicted_delivery_days").to_csv(OUT/"shipping_mode_optimization_example.csv",index=False)
    pd.DataFrame(confusion_matrix(ycte,pc),index=["Actual Late","Actual On-Time"],columns=["Predicted Late","Predicted On-Time"]).to_csv(OUT/"confusion_matrix.csv")
    plt.figure(figsize=(7,5)); plt.scatter(yrte,pr,alpha=.45,s=18); lo=min(yrte.min(),pr.min()); hi=max(yrte.max(),pr.max()); plt.plot([lo,hi],[lo,hi],"--"); plt.xlabel("Actual Delivery Days"); plt.ylabel("Predicted Delivery Days"); plt.title("Actual vs Predicted Delivery Time"); plt.tight_layout(); plt.savefig(OUT/"actual_vs_predicted_delivery.png",dpi=150); plt.close()
    print(f"Example predicted delivery: {float(example_out.predicted_delivery_days.iloc[0]):.2f} days")
    print("Outputs saved to output/")
if __name__=="__main__": main()
