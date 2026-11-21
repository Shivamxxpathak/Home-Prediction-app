import os
from pathlib import Path
from flask import Flask,render_template,request,jsonify
import joblib
from model import create_and_train_model,prepare_features,GLOBAL_REGIONS

ROOT=Path(__file__).resolve().parent
MODEL_PATH=ROOT/"model.pkl"
app=Flask(__name__,template_folder="templates",static_folder="static")

def load_model():
    try:
        saved=joblib.load(MODEL_PATH)
        if isinstance(saved,dict) and "model" in saved and "feature_columns" in saved: return saved
    except (FileNotFoundError,EOFError,ValueError): pass
    app.logger.info("Training model because no compatible artifact was found.")
    return create_and_train_model()

model_data=load_model()
model=model_data.get("model") if model_data else None

@app.route("/")
def home():
    return render_template("index.html",regions=GLOBAL_REGIONS,metrics=model_data.get("metrics",{}) if model_data else {})

@app.route("/health")
def health():
    return jsonify({"status":"ok" if model is not None else "model_unavailable","model_loaded":model is not None,
                    "selected_model":model_data.get("metrics",{}).get("selected_model") if model_data else None})

@app.route("/metrics")
def metrics():
    if not model_data: return jsonify({"success":False,"error":"Metrics unavailable."}),503
    return jsonify({"success":True,**model_data.get("metrics",{})})

@app.route("/predict",methods=["POST"])
def predict():
    if model is None: return jsonify({"success":False,"error":"Model unavailable."}),503
    try:
        data=request.get_json(silent=True) or {}
        area=float(data.get("area")); bedrooms=int(data.get("bedrooms")); age=float(data.get("age"))
        region=data.get("region","North America")
        if not 700<=area<=6000: raise ValueError("Area must be between 700 and 6,000 sq ft.")
        if not 1<=bedrooms<=7: raise ValueError("Bedrooms must be between 1 and 7.")
        if not 0<=age<=69: raise ValueError("Age must be between 0 and 69 years.")
        if region not in GLOBAL_REGIONS: raise ValueError("Select a valid market region.")
        value=max(0.0,float(model.predict(prepare_features(area,bedrooms,age,region))[0]))
        return jsonify({"success":True,"prediction":f"${value:,.0f}","prediction_numeric":round(value,2),
                        "region":region,"model":model_data.get("metrics",{}).get("selected_model","Regression model"),
                        "disclaimer":"Educational estimate, not a professional property valuation."})
    except (TypeError,ValueError) as exc: return jsonify({"success":False,"error":str(exc)}),400
    except Exception:
        app.logger.exception("Prediction request failed")
        return jsonify({"success":False,"error":"Prediction failed. Check server logs."}),500

if __name__=="__main__": app.run(debug=False,host="0.0.0.0",port=int(os.environ.get("PORT",5000)))
