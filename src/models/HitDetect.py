import torch
import torch.nn as nn
import numpy as np


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

        # for storing the prediction result
        self.preds = []
        self.hit_frames = {}

    def reset(self):
        self.preds = []
        self.hit_frames = {}

    def setup_HitDetect(self):
        self.__hitdetect = torch.load("src/models/weights/hitdetect.pth")
        self.__hitdetect.to(self.device).eval()

    def del_HitDetect(self):
        del self.__hitdetect

    def get_hits_event(self, top_kp, bottom_kp, court_kp, ball_pos):
        pass
