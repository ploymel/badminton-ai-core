import sys
import pandas as pd
import os


sys.path.append("src/tools")
sys.path.append("src/models")
from utils import read_json, write_json, extract_numbers
from HitDetect import HitDetector, HitModel


def convert_input2dataframe(shuttle, players_kp, court_kp):
    input_data = {
        "frame": [],
        "top": [],
        "bottom": [],
        "court": [],
        "ball": [],
        "net": [],
    }
    # convert to dataframe
    for frame in shuttle.keys():
        try:
            players_info = players_kp[frame]
        except:
            continue

        if players_info["top"] is None or players_info["bottom"] is None:
            continue
        if shuttle[frame]["visible"] == 0:
            continue
        input_data["frame"].append(frame)
        input_data["top"].append(players_info["top"])
        input_data["bottom"].append(players_info["bottom"])
        input_data["ball"].append([shuttle[frame]["x"], shuttle[frame]["y"]])
        input_data["court"].append(court_kp["court_info"])
        input_data["net"].append(court_kp["net_info"])

    return pd.DataFrame(input_data)


def event_detect(json_path, players_kp_path, court_kp_path, result_path):
    loca_dict = read_json(json_path)
    court_kp = read_json(court_kp_path)
    players_kp = read_json(players_kp_path)

    # start the model
    hit_detect = HitDetector()

    # convert input data to dataframe
    hits_data = convert_input2dataframe(loca_dict, players_kp, court_kp)

    # get hit info
    result, result_fallback = hit_detect.get_hits_event(hits_data, fps=30)

    # copy hits data for fallback's model results
    hits_data_fallback = hits_data.copy()
    hits_data["pred"] = result + [0] * (len(hits_data) - len(result))
    hits_data_fallback["pred"] = result_fallback + [0] * (
        len(hits_data_fallback) - len(result_fallback)
    )

    # Identify rows to keep based on changes in the 'pred' value or the last row of the DataFrame
    # The approach uses a combination of `shift()` for comparison and handling edge cases
    hits_data["keep"] = (hits_data["pred"] != hits_data["pred"].shift(-1)) | (
        hits_data.index == len(hits_data) - 1
    )
    hits_data_fallback["keep"] = (
        hits_data_fallback["pred"] != hits_data_fallback["pred"].shift(-1)
    ) | (hits_data_fallback.index == len(hits_data_fallback) - 1)

    # Filter rows to keep
    hits_data = hits_data[hits_data["keep"]].drop(
        "keep", axis=1
    )  # Drop the temporary 'keep' column
    hits_data_fallback = hits_data_fallback[hits_data_fallback["keep"]].drop(
        "keep", axis=1
    )  # Drop the temporary 'keep' column (for fallback)

    # drop prediction = 0
    hits_data = hits_data.loc[hits_data["pred"] != 0]
    hits_data_fallback = hits_data_fallback[hits_data_fallback["pred"] != 0]

    # save
    json_name = os.path.splitext(os.path.basename(json_path))[0]
    folder_name, _ = extract_numbers(json_name)
    event_path = os.path.join(result_path, f"event/{folder_name}")
    event_fallback_path = os.path.join(result_path, f"event_fallback/{folder_name}")

    os.makedirs(event_path, exist_ok=True)
    os.makedirs(event_fallback_path, exist_ok=True)

    # write json
    write_json(
        hits_data.to_dict(orient="records"), json_name, f"{event_path}", mode="w"
    )

    write_json(
        hits_data_fallback.to_dict(orient="records"),
        json_name,
        f"{event_fallback_path}",
        mode="w",
    )
