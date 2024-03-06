import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import glob


class ShotTypeModel(nn.Module):
    """
    Shot type detection model
    """

    def __init__(self, feature_dim, num_consecutive_frames):
        super(ShotTypeModel, self).__init__()
        self.num_consecutive_frames = num_consecutive_frames
        self.feature_dim = feature_dim

        self.gru1 = nn.GRU(
            feature_dim // num_consecutive_frames,
            64,
            bidirectional=True,
            batch_first=True,
        )
        self.gru2 = nn.GRU(128, 64, bidirectional=True, batch_first=True)
        self.global_maxpool = nn.MaxPool1d(num_consecutive_frames)
        self.dense = nn.Linear(128, 1)  # Output layer for binary classification

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.float()
        x = x.view(
            batch_size,
            self.num_consecutive_frames,
            self.feature_dim // self.num_consecutive_frames,
        )
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        x = x.transpose(1, 2)
        x = self.global_maxpool(x).squeeze()
        x = self.dense(x)
        return x  # Removed softmax, output is now logits


class ShotDetect(object):
    """
    For predicting which shot has been hit
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_consecutive_frames = 30
        self.normalization = True

        self.setup_ShotDetect()

    def setup_ShotDetect(self):
        self.__shot_detect_models = dict()
        for model_path in glob.glob("src/models/weights/shot_detect/*.pth"):
            _, model_name = os.path.split(model_path)
            model_name = model_name.replace(".pth", "")
            __shot_detect = (
                torch.load(model_path, map_location=self.device).to(self.device).eval()
            )

            self.__shot_detect_models[model_name] = __shot_detect

    def del_ShotDetect(self):
        del self.__shot_detect_models

    def get_shots_info(self, data):
        rows = len(data)
        # Determine if padding is needed
        remainder = rows % 12
        if remainder > 0:
            num_to_pad = 12 - remainder
        else:
            num_to_pad = 0
        # print("Padding needs: ", remainder)

        # Pad the dataframe if necessary
        if num_to_pad > 0:
            last_row = data.iloc[-1]
            padding_data = np.tile(last_row.values, (num_to_pad, 1))
            padded_data = pd.DataFrame(padding_data, columns=data.columns)
            data = pd.concat([data, padded_data], axis=0)
            data = data.reset_index(drop=True)

        shot_lists = {model_name: [] for model_name in self.__shot_detect_models.keys()}
        for i in range(len(data)):
            # Won't do the rest (seeing num_consecutive_frames ahead)
            if i >= len(data) - self.num_consecutive_frames:
                break

            # Prepare data for prediction (current_frame, ..., current_frame+num_consecutive_frames)
            ori_data = data.loc[i : i + self.num_consecutive_frames - 1, :].copy()
            ori_data = ori_data.reset_index(drop=True)

            input_data = []
            for index, row in ori_data.iterrows():
                # Pre-processing data
                top = np.array(row["top"]).reshape(-1, 2)
                bottom = np.array(row["bottom"]).reshape(-1, 2)
                court = np.array(row["court"]).reshape(-1, 2)
                ball = np.array(row["ball"]).reshape(-1, 2)
                net = np.array(row["net"]).reshape(-1, 2)

                frame_data = np.concatenate((top, bottom, court, net, ball), axis=0)

                if self.normalization:
                    # Map the x-coordinate from 0 to 1920 to 1 to 2
                    frame_data[:, 0] /= 1920

                    # Map the y-coordinate from 0 to 1080 to 1 to 2
                    frame_data[:, 1] /= 1080

                input_data.append(frame_data.reshape(1, -1))

            # Predict the outcome
            input_data = np.array(input_data).reshape(1, -1)
            # Loop through all available models
            for model_name, model in self.__shot_detect_models.items():
                with torch.no_grad():
                    outputs = model(
                        torch.FloatTensor(input_data).to(self.device)
                    ).squeeze()

                shot_lists[model_name].append(outputs)

        return shot_lists
