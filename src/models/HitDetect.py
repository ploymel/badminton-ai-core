import torch
import torch.nn as nn
import numpy as np
import pandas as pd


class HitModel(nn.Module):
    """
    Hit detector model
    """

    def __init__(self, feature_dim, num_consecutive_frames):
        super(HitModel, self).__init__()
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
        self.dense = nn.Linear(128, 3)
        self.softmax = nn.Softmax(dim=-1)

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
        x = self.softmax(x)
        return x


class HitDetector(object):
    """
    For predicting which frame the hit occurs
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_consecutive_frames = 12
        self.normalization = True
        self.fps = 30

        # for optimization
        self.optim_num_consecutive_frames = 5

        self.setup_HitDetect()

    def setup_HitDetect(self):
        self.__hitdetect = torch.load(
            "src/models/weights/hit_detectv2.pth", map_location=self.device
        )
        self.__hitdetect.to(self.device).eval()

    def del_HitDetect(self):
        del self.__hitdetect

    def get_hits_event(self, data, fps):
        rows = len(data)
        # Determine if padding is needed
        remainder = rows % 12
        if remainder > 0:
            num_to_pad = 12 - remainder
        else:
            num_to_pad = 0
        print("Padding needs: ", remainder)

        # Pad the dataframe if necessary
        if num_to_pad > 0:
            last_row = data.iloc[-1]
            padding_data = np.tile(last_row.values, (num_to_pad, 1))
            padded_data = pd.DataFrame(padding_data, columns=data.columns)
            data = pd.concat([data, padded_data], axis=0)
            data = data.reset_index(drop=True)

        hit_lists = []
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

                frame_data = np.concatenate((top, bottom, court, ball), axis=0)

                if self.normalization:
                    # Map the x-coordinate from 0 to 1920 to 1 to 2
                    frame_data[:, 0] /= 1920

                    # Map the y-coordinate from 0 to 1080 to 1 to 2
                    frame_data[:, 1] /= 1080

                input_data.append(frame_data.reshape(1, -1))

            # Predict the outcome
            input_data = np.array(input_data).reshape(1, -1)
            with torch.no_grad():
                outputs = self.__hitdetect(
                    torch.FloatTensor(input_data).to(self.device)
                )

            pred = torch.argmax(outputs).item()
            hit_lists.append(pred)

        # optimize the hit_lists
        optim_hit_lists = self.__optimize_final_list_corrected(
            hit_lists, fps, self.optim_num_consecutive_frames
        )

        return optim_hit_lists

    def __optimize_final_list_corrected(
        self, final_list, fps=30, num_consecutive_frames=6
    ):
        optimized_list = (
            final_list.copy()
        )  # Work on a copy of the list to keep original intact

        # Helper function to apply the optimization rule
        def apply_optimization(start, end):
            for j in range(start, end - 1):  # Change all but the last to 0
                optimized_list[j] = 0

        i = 0
        while i < len(final_list):
            if final_list[i] in [1, 2]:  # Check if the current element is either 1 or 2
                start = i
                # Move forward as long as the next element is the same as the current one
                while (
                    i + 1 < len(final_list) and final_list[i + 1] == final_list[start]
                ):
                    i += 1
                end = i + 1  # Mark the end of the sequence
                sequence_length = end - start
                # Check if the sequence is long enough and consists of the same number (either all 1s or all 2s)
                if sequence_length >= num_consecutive_frames:
                    apply_optimization(start, end)  # Apply optimization to the sequence
                elif (
                    sequence_length < num_consecutive_frames
                ):  # If the sequence is shorter than 6
                    for j in range(
                        start, end
                    ):  # Convert all elements of this sequence to 0
                        optimized_list[j] = 0
            i += 1

        # Now enforce rally hit rules
        last_hit = None
        for i in range(1, len(optimized_list)):
            if optimized_list[i] in [1, 2]:
                # The paper said 0.5*fps but for the net drop i think it can be less than that so, i use 0.3 instead
                if last_hit is not None and (
                    optimized_list[i] == last_hit or i - last_hit_index <= fps * 0.3
                ):
                    optimized_list[i] = 0  # Rule violation: too close or same player
                else:
                    last_hit = optimized_list[i]
                    last_hit_index = i

        return optimized_list
