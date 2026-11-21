"""Reproducible housing-price training and evaluation pipeline."""
from pathlib import Path
import argparse, json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"
MODEL_PATH=ROOT/"model.pkl"
METRICS_PATH=ROOT/"reports"/"model_metrics.json"
PLOTS=ROOT/"reports"/"plots"
GLOBAL_REGIONS=["North America","Europe","Asia","Latin America","Africa","Oceania"]
REGION_MULTIPLIERS={"North America":1.0,"Europe":.95,"Asia":.8,"Latin America":.6,"Africa":.45,"Oceania":1.05}
FEATURE_COLUMNS=["Area","Bedrooms","Age"]+[f"Region_{r}" for r in GLOBAL_REGIONS]
REQUIRED=["Area","Bedrooms","Age","Region","Price"]

def generate_synthetic_dataset(n_samples=2000,random_state=42):
    rng=np.random.RandomState(random_state)
    area=rng.randint(700,6000,n_samples); bedrooms=rng.randint(1,8,n_samples)
    age=rng.randint(0,70,n_samples); regions=rng.choice(GLOBAL_REGIONS,n_samples)
    price=(30000+area*130+bedrooms*12000-age*800)*np.array([REGION_MULTIPLIERS[r] for r in regions])
    price+=rng.normal(0,20000,n_samples)
    return pd.DataFrame({"Area":area,"Bedrooms":bedrooms,"Age":age,"Region":regions,"Price":np.maximum(price,1000)})

def load_dataset(csv_path=None):
    path=Path(csv_path) if csv_path else DATA/"housing.csv"
    if path.exists(): df=pd.read_csv(path); source=f"CSV: {path.as_posix()}"
    else: df=generate_synthetic_dataset(); source="Synthetic educational dataset; generated, not real market data"
    missing=set(REQUIRED)-set(df.columns)
    if missing: raise ValueError(f"Missing required columns: {sorted(missing)}")
    df=df[REQUIRED].copy()
    for c in ["Area","Bedrooms","Age","Price"]: df[c]=pd.to_numeric(df[c],errors="coerce")
    df["Region"]=df["Region"].astype("string").str.strip()
    df=df.dropna(subset=REQUIRED).drop_duplicates()
    df=df[df.Region.isin(GLOBAL_REGIONS)]
    df=df[(df.Area>0)&(df.Bedrooms>0)&(df.Age>=0)&(df.Price>0)]
    if len(df)<30: raise ValueError("At least 30 valid unique rows are required.")
    return df,source

def prepare_features(area,bedrooms,age,region):
    if region not in GLOBAL_REGIONS: raise ValueError("Select a valid market region.")
    row=pd.DataFrame([{"Area":area,"Bedrooms":bedrooms,"Age":age,"Region":region}])
    return pd.get_dummies(row,columns=["Region"]).reindex(columns=FEATURE_COLUMNS,fill_value=0)

def save_plots(y,pred,importance=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        PLOTS.mkdir(parents=True,exist_ok=True)
        fig,ax=plt.subplots(figsize=(7,6)); ax.scatter(y,pred,alpha=.65)
        bounds=[min(float(y.min()),float(np.min(pred))),max(float(y.max()),float(np.max(pred)))]
        ax.plot(bounds,bounds,"--"); ax.set(xlabel="Actual price",ylabel="Predicted price",title="Actual vs predicted")
        fig.tight_layout(); fig.savefig(PLOTS/"actual_vs_predicted.png",dpi=160); plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,5)); ax.scatter(pred,np.asarray(y)-np.asarray(pred),alpha=.65)
        ax.axhline(0,linestyle="--"); ax.set(xlabel="Predicted price",ylabel="Residual",title="Residual analysis")
        fig.tight_layout(); fig.savefig(PLOTS/"residual_plot.png",dpi=160); plt.close(fig)
        if importance is not None:
            s=pd.Series(importance,index=FEATURE_COLUMNS).sort_values().tail(10)
            fig,ax=plt.subplots(figsize=(8,5)); s.plot(kind="barh",ax=ax)
            ax.set(title="Feature importance",xlabel="Importance")
            fig.tight_layout(); fig.savefig(PLOTS/"feature_importance.png",dpi=160); plt.close(fig)
    except ImportError: print("Matplotlib unavailable; plots skipped.")

def create_and_train_model(csv_path=None):
    df,source=load_dataset(csv_path)
    X=pd.get_dummies(df[["Area","Bedrooms","Age","Region"]],columns=["Region"]).reindex(columns=FEATURE_COLUMNS,fill_value=0)
    y=df.Price.astype(float)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)
    candidates={
        "Linear Regression":LinearRegression(),
        "Random Forest":RandomForestRegressor(n_estimators=250,min_samples_leaf=2,random_state=42,n_jobs=-1),
        "HistGradientBoosting":HistGradientBoostingRegressor(max_iter=200,learning_rate=.08,random_state=42)}
    fitted={}; scores={}
    for name,est in candidates.items():
        est.fit(X_train,y_train); pred=est.predict(X_test); fitted[name]=est
        scores[name]={"MAE":float(mean_absolute_error(y_test,pred)),
                      "RMSE":float(np.sqrt(mean_squared_error(y_test,pred))),
                      "R2":float(r2_score(y_test,pred))}
    winner=min(scores,key=lambda n:scores[n]["RMSE"]); model=fitted[winner]; pred=model.predict(X_test)
    save_plots(y_test,pred,getattr(model,"feature_importances_",None))
    metrics={"dataset_source":source,"rows_after_cleaning":len(df),"train_rows":len(X_train),
             "test_rows":len(X_test),"split":"80/20; random_state=42","selected_model":winner,"models":scores,
             "note":"Synthetic metrics demonstrate the pipeline only; they do not establish real-world valuation accuracy." if "Synthetic" in source else "Held-out test metrics. Confirm dataset provenance and representativeness."}
    bundle={"model":model,"feature_columns":FEATURE_COLUMNS,"regions":GLOBAL_REGIONS,"metrics":metrics,"data_source":source}
    joblib.dump(bundle,MODEL_PATH); METRICS_PATH.parent.mkdir(parents=True,exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print(json.dumps(metrics,indent=2)); return bundle

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",help="CSV with Area,Bedrooms,Age,Region,Price columns")
    create_and_train_model(parser.parse_args().data)
