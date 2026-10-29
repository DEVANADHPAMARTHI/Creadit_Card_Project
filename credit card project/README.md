# Credit Card Customer Classification

A machine-learning project that predicts whether a credit-card customer is classified as a **Good Customer** or **Bad Customer**. It includes a Flask web interface, saved model artifacts, model-training scripts, and Jupyter notebooks for data preparation and exploration.

> **Disclaimer:** This project is for educational purposes only. Its predictions are not financial advice and should not be used to make real credit, lending, or other high-impact decisions.

## Features

- Flask web form for submitting customer attributes and viewing a model prediction.
- Pre-trained classifier and feature scaler loaded by the application.
- Python scripts and notebooks covering data preparation, feature selection, transformations, and model experiments.

## Requirements

- Python 3.11 or later
- Packages listed in [`requirements.txt`](requirements.txt)

## Setup and run

From the project directory, create and activate a virtual environment, install the pinned dependencies, and start Flask:

### Windows (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000> in your browser. The app expects `Credit_card_Model.pkl` and `scaled_model.pkl` in the project directory.

## Prediction inputs

The form accepts 12 numeric values: age, open credit lines and loans, real-estate loans or lines, monthly income, gender encoding, four region indicators, housing encoding, occupation encoding, and education encoding. Use the same feature definitions and category encodings that were used to train the included model. For one-hot encoded categories, use the expected 0/1 indicator values.

## Project structure

| File or directory | Purpose |
| --- | --- |
| `app.py` | Flask application and prediction endpoint |
| `templates/index.html` | Prediction form |
| `Credit_card_Model.pkl`, `scaled_model.pkl` | Trained classifier and scaler |
| `main.py`, `all_models.py`, `fs.py`, and other Python scripts | Data preparation, feature selection, and model experiments |
| `*.ipynb` | Data cleaning, encoding, feature selection, and transformation notebooks |
| `CREDIT DATA DICTIONARY.xlsx` | Dataset feature reference |
| `requirements.txt` | Pinned Python dependencies |
| `Procfile` | Process declaration for compatible deployment platforms |

The training dataset (`creditcard.csv`) is not included. The training scripts require an authorized copy in the project directory. Check the dataset's terms before use or redistribution.

## Security note

The application loads Python pickle model files. Only use model files from a trusted source; loading an untrusted pickle can execute arbitrary code.
