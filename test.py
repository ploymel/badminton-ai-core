loca_json_path = "res/ball/loca_info/set1/set1_1542-2189.json"
court_kp_path = "res/courts/court_kp/set1.json"
players_kp_path = "res/players/player_kp/set1.json"

from src.tools.utils import read_json
from src.models.HitDetect import HitDetector, HitModel
from src.models.ShotDetect import ShotTypeModel, ShotDetect
import pandas as pd


def event_ml_detection():
    shuttle = read_json(loca_json_path)
    court_kp = read_json(court_kp_path)
    players_kp = read_json(players_kp_path)

    input_data = {
        "frame": [],
        "top": [],
        "bottom": [],
        "court": [],
        "ball": [],
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

    input_data = pd.DataFrame(input_data)

    # start the model
    hit_detect = HitDetector()
    shot_detect = ShotDetect()

    # get hit info
    result, result_fallback = hit_detect.get_hits_event(input_data, fps=30)
    # get shot types info
    shot_type_results = shot_detect.get_shots_info(input_data)
    # print(shot_type_results)

    input_data["pred"] = result + [0] * (len(input_data) - len(result))
    input_data["shot_type"] = shot_type_results + [0] * (
        len(input_data) - len(shot_type_results)
    )
    # input_data["pred_fallback"] = result_fallback + [0] * (
    #     len(input_data) - len(result_fallback)
    # )

    # Identify rows to keep based on changes in the 'pred' value or the last row of the DataFrame
    # The approach uses a combination of `shift()` for comparison and handling edge cases
    input_data["keep"] = (input_data["pred"] != input_data["pred"].shift(-1)) | (
        input_data.index == len(input_data) - 1
    )
    # input_data["keep_fallback"] = (
    #     input_data["pred_fallback"] != input_data["pred_fallback"].shift(-1)
    # ) | (input_data.index == len(input_data) - 1)

    # Filter rows to keep
    filtered_data = input_data[input_data["keep"]].drop(
        "keep", axis=1
    )  # Drop the temporary 'keep' column
    # filtered_data = input_data[input_data["keep_fallback"]].drop(
    #     "keep_fallback", axis=1
    # )  # Drop the temporary 'keep_fallback' column
    filtered_data = filtered_data[filtered_data["pred"] != 0]
    print(filtered_data)


event_ml_detection()
