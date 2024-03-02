from ultralytics import YOLO
import sys
import cv2
import numpy as np
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

sys.path.append("src/models")
sys.path.append("src/models")

from utils import read_json


def highest_confidence_box_per_class(boxes, clss, confs):
    unique_classes = set(clss)
    highest_conf_boxes = []
    for cls in unique_classes:
        cls_indices = [i for i, x in enumerate(clss) if x == cls]
        # For each class, find the box with the highest confidence
        highest_conf_index = max(cls_indices, key=lambda i: confs[i])
        highest_conf_boxes.append(
            (cls, boxes[highest_conf_index], confs[highest_conf_index])
        )
    return highest_conf_boxes


def iou_and_inside_check(boxA, boxB):
    """
    Check if boxA is inside boxB and calculate the Intersection Over Union (IoU).
    Return True if boxA is significantly inside boxB and IoU is above a threshold.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])

    # Consider a box significantly inside another if the intersection area is most of its area, say 70%
    is_inside = interArea / boxAArea > 0.7
    return is_inside


def filter_boxes(boxes, clss, confs):
    # Assuming boxes, clss, confs are available
    highest_conf_boxes = highest_confidence_box_per_class(boxes, clss, confs)

    # Find the box of class 0
    box_class_0 = next((box for cls, box, conf in highest_conf_boxes if cls == 0), None)

    # Filter boxes of class 1 and 2 based on being inside box of class 0
    final_boxes = [(0.0, box_class_0)]
    for cls, box, conf in highest_conf_boxes:
        if cls in [1, 2] and box_class_0:
            if iou_and_inside_check(box, box_class_0):
                final_boxes.append((cls, box, conf))

    return final_boxes


def crop_score(image, bbox):
    """
    Crop Score
    """
    x1, y1 = bbox[0], bbox[1]
    x2, y2 = bbox[2], bbox[3]

    cropped_image = image.copy()
    cropped_image = cropped_image[int(y1) : int(y2), int(x1) : int(x2)]

    return cropped_image


class ScoreDetect(object):
    """
    A class uses for detecting and reading scores from score box
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.normal_scorebox_info = None
        self.got_info = False
        self.mse = None
        self.fixed_scorebox = None
        self.setup_YOLO()
        self.setup_trOCR()

    def reset(self):
        self.got_info = False
        self.normal_scorebox_info = None
        self.fixed_scorebox = None

    def setup_YOLO(self):
        self.___scorebox_detect_YOLO = YOLO(
            "src/models/weights/scorebox_detect_yolov8s.pt"
        )

    def setup_trOCR(self):
        self.__score_reader_processor = TrOCRProcessor.from_pretrained(
            "src/models/weights/score_reader_trocr"
        )
        self.__score_reader_trOCR = (
            VisionEncoderDecoderModel.from_pretrained(
                "src/models/weights/score_reader_trocr"
            )
            .to(self.device)
            .eval()
        )

    def del_YOLO(self):
        del self.___scorebox_detect_YOLO

    def del_trOCR(self):
        del self.__score_reader_trOCR

    def pre_process(self, video_path):
        # Open the video file
        video = cv2.VideoCapture(video_path)

        # Get video properties
        fps = video.get(cv2.CAP_PROP_FPS)

        # Number of consecutively detected pitch frames
        last_count = 0
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        scorebox_info_list = []
        # the number of skip frames per time
        skip_frames = max(int(fps) // 5, 5)

        while True:
            # Read a frame from the video
            current_frame = int(video.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frame = video.read()

            print(
                f"video is pre-processing for scorebox, current frame is {current_frame}"
            )

            # If there are no more frames, break the loop
            if last_count >= skip_frames:
                self.normal_scorebox_info = scorebox_info_list[skip_frames // 2]
                for scorebox_info in scorebox_info_list:
                    if not self.__check_scorebox(scorebox_info):
                        self.normal_scorebox_info = None
                        scorebox_info = []
                        last_count = 0
                        print("Detect the wrong scorebox!")
                        break

                if self.normal_scorebox_info is not None:
                    return max(0, current_frame - 2 * skip_frames)
                else:
                    continue

            if not ret:
                # release the video
                video.release()
                return max(0, current_frame - 2 * skip_frames)

            scorebox_info, have_scorebox = self.get_score_info(frame)
            if have_scorebox:
                last_count += 1
                scorebox_info_list.append(scorebox_info["scorebox"]["box"])
            else:
                if current_frame + skip_frames >= total_frames:
                    print("Fail to pre-process! Please go check the video or program!")
                    exit(0)

                video.set(cv2.CAP_PROP_POS_FRAMES, current_frame + skip_frames)
                last_count = 0
                scorebox_info_list = []

    def __check_scorebox(self, scorebox_info):
        vec1 = np.array(self.normal_scorebox_info)
        vec2 = np.array(scorebox_info)
        mse = np.square(vec1 - vec2).mean()
        self.mse = mse
        if mse > 100:
            return False
        return True

    def __check_isvalid_score(self, scorebox):
        if (
            scorebox["scorebox_top"]["score"] == 0
            and scorebox["scorebox_buttom"]["score"] == 1
        ):
            return True
        if (
            scorebox["scorebox_top"]["score"] == 1
            and scorebox["scorebox_buttom"]["score"] == 0
        ):
            return True
        return False

    def get_score_info(self, img):
        image = img.copy()
        self.mse = None

        output = self.___scorebox_detect_YOLO(image, verbose=False)
        # Post-process boxes
        boxes = output[0].boxes.xyxy.cpu().tolist()

        if len(boxes) >= 3:
            # Extract prediction output
            clss = output[0].boxes.cls.cpu().tolist()
            confs = output[0].boxes.conf.float().cpu().tolist()

            final_boxes = filter_boxes(boxes, clss, confs)

            self.__correct_points = self.__correction(final_boxes)

            # Check whether the output contains all scoreboxes or not
            if len(final_boxes) == 3:
                self.__true_scorebox_points = final_boxes[0][-1]
                # check if its value is normal
                if self.normal_scorebox_info is not None:
                    self.got_info = self.__check_scorebox(self.__true_scorebox_points)
                    if not self.got_info:
                        return None, self.got_info

                self.got_info = True

                # If there is a fixed scorebox, use that
                if self.fixed_scorebox is not None:
                    # Crop scores
                    top_score_img = crop_score(
                        img, self.fixed_scorebox["scorebox_top"]["box"]
                    )
                    bottom_score_img = crop_score(
                        img, self.fixed_scorebox["scorebox_buttom"]["box"]
                    )
                else:
                    # Crop scores
                    top_score_img = crop_score(
                        img, self.__correct_points["scorebox_top"]["box"]
                    )
                    bottom_score_img = crop_score(
                        img, self.__correct_points["scorebox_buttom"]["box"]
                    )

                # Preprocess input images
                pixel_values = self.__score_reader_processor(
                    images=[top_score_img, bottom_score_img], return_tensors="pt"
                ).pixel_values
                pixel_values = pixel_values.to(self.device)

                # Run recognition
                generated_ids = self.__score_reader_trOCR.generate(
                    pixel_values, max_new_tokens=2
                )
                generated_text = self.__score_reader_processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )

                try:
                    self.__correct_points["scorebox_top"]["score"] = int(
                        generated_text[0]
                    )
                    self.__correct_points["scorebox_buttom"]["score"] = int(
                        generated_text[1]
                    )

                    if self.fixed_scorebox is not None:
                        if self.__check_isvalid_score(self.__correct_points):
                            self.fixed_scorebox = self.__correct_points

                except:
                    self.got_info = False
                    return None, self.got_info

                return self.__correct_points, self.got_info

            else:
                self.got_info = False
                return None, self.got_info

        self.got_info = False
        return None, self.got_info

    def __correction(self, final_boxes):
        classes = {0: "scorebox", 1: "scorebox_buttom", 2: "scorebox_top"}
        correction = dict()
        for box in final_boxes:
            correction[classes[int(box[0])]] = {"box": box[1]}

        return correction
