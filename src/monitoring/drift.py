from pathlib import Path
import pandas as pd
from src.db import get_split_data
from evidently import Report
from evidently.presets import DataDriftPreset

DARK_COLORS = {"Black", "Navy", "Brown", "Grey"}
LIGHT_COLORS = {"Purple", "Orange", "Pink", "Yellow"}

def run_drift():
    df_x, df_y = get_split_data(split="val")

    df = df_x.merge(df_y, on="product_id")
    df["color_list"] = df["color_tags"].str.split(",")

    ref_df = df[df["color_list"].apply(lambda tags: bool(DARK_COLORS & set(tags)))]
    curr_df = df[df["color_list"].apply(lambda tags: bool(LIGHT_COLORS & set(tags)))]

    all_colors = sorted(DARK_COLORS | LIGHT_COLORS)

    ref_encoded = pd.DataFrame(
        [{c: int(c in tags) for c in all_colors} for tags in ref_df["color_list"]]
    )
    curr_encoded = pd.DataFrame(
        [{c: int(c in tags) for c in all_colors} for tags in curr_df["color_list"]]
    )

    datadrift_dataset_report = Report(metrics=[DataDriftPreset()])

    snapshot = datadrift_dataset_report.run(current_data=curr_encoded, reference_data=ref_encoded)

    reports_dir = Path(__file__).resolve().parents[2] / "reports"

    with open(reports_dir / "data_drift_report.json", "w") as f:
        f.write(snapshot.json())

    snapshot.save_html(str(reports_dir / "data_drift_report.html"))

if __name__ == "__main__":
    run_drift()
