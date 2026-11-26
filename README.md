# Ethiopia Drought Risk Intelligence

Interactive Streamlit dashboard that predicts drought risk (No / Moderate / Severe) from climate
inputs and recommends crops and field actions.

## Repository layout
```
ethi-drought-prediction/
├── app.py
├── requirements.txt
├── README.md
├── .streamlit/config.toml
├── models/
│   ├── drought_model.pkl
│   └── metadata.json
└── data/
    └── dashboard_data.csv
```

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
Streamlit Community Cloud -> New app -> select this repo, branch `main`, main file `app.py`
(or `ethi-drought-prediction/app.py` if it sits in a sub-folder). Under Advanced settings choose Python 3.12.
