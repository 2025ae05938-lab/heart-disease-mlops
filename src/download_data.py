"""Step 1a: Download the UCI Heart Disease (Cleveland) dataset.

Run from the project root:
    python src/download_data.py
"""
import os  # os lets us create folders and build file paths
import urllib.request  # built-in module for downloading files from a URL

# Direct link to the raw Cleveland data file on the UCI repository
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

# The raw file has no header row, so we define the 14 column names ourselves
COLUMNS = [
    "age",       # age in years
    "sex",       # 1 = male, 0 = female
    "cp",        # chest pain type (1-4)
    "trestbps",  # resting blood pressure (mm Hg)
    "chol",      # serum cholesterol (mg/dl)
    "fbs",       # fasting blood sugar > 120 mg/dl (1 = true)
    "restecg",   # resting ECG result (0-2)
    "thalach",   # maximum heart rate achieved
    "exang",     # exercise-induced angina (1 = yes)
    "oldpeak",   # ST depression induced by exercise
    "slope",     # slope of the peak exercise ST segment
    "ca",        # number of major vessels (0-3) coloured by fluoroscopy
    "thal",      # thalassemia (3 = normal, 6 = fixed defect, 7 = reversible)
    "target",    # 0 = no heart disease, 1-4 = heart disease
]

RAW_PATH = os.path.join("data", "heart_raw.csv")  # where the raw file is saved


def download(url: str = DATA_URL, out_path: str = RAW_PATH) -> str:
    """Download the dataset and save it as a CSV with a header row."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)  # make data/ if missing
    raw_text_path = out_path + ".tmp"  # temporary file for the raw download
    try:
        urllib.request.urlretrieve(url, raw_text_path)  # fetch the file
    except Exception as exc:  # no internet, blocked network, link moved...
        raise SystemExit(
            f"Download failed ({exc}).\n"
            "Manual fallback: open https://archive.ics.uci.edu/dataset/45/heart+disease, "
            "download the zip, extract 'processed.cleveland.data' and save it as "
            f"'{raw_text_path}', then run this script again."
        )
    with open(raw_text_path, "r") as src, open(out_path, "w") as dst:
        dst.write(",".join(COLUMNS) + "\n")  # write the header line first
        dst.write(src.read())  # then copy the raw data rows underneath
    os.remove(raw_text_path)  # delete the temporary file
    print(f"Saved dataset to {out_path}")
    return out_path


if __name__ == "__main__":  # only runs when you execute this file directly
    download()
