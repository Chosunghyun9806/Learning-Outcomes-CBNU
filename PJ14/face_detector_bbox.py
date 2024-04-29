import multiprocessing
import os
import math
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Tuple, Union


def _normalized_to_pixel_coordinates(normalized_x: float, normalized_y: float,
                                     image_width: int, image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                          math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)

    return x_px, y_px


class FaceDetector_BBox(multiprocessing.Process):

    def __init__(self, input_queue: multiprocessing.Queue, output_queue: multiprocessing.Queue, model_path: str):
        super(FaceDetector_BBox, self).__init__()

        self.MARGIN = 10
        self.ROW_SIZE = 10
        self.FONT_SIZE = 1
        self.FONT_THICKNESS = 1
        self.TEXT_COLOR = (255, 0, 0)

        self.input_queue = input_queue
        self.output_queue = output_queue
        self.model_path = model_path

    def visualize(self, image, detection_result) -> np.ndarray:
        """Draws bounding boxes and keypoints on the input image and return it.
        Args:
          image: The input RGB image.
          detection_result: The list of all "Detection" entities to be visualize.
        Returns:
          Image with bounding boxes.
        """
        annotated_image = image.copy()
        height, width, _ = image.shape

        for detection in detection_result.detections:
            # Draw bounding_box
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(annotated_image, start_point, end_point, self.TEXT_COLOR, 3)

            # Draw keypoints
            for keypoint in detection.keypoints:
                keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                               width, height)
                color, thickness, radius = (0, 255, 0), 1, 2
                cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

            # Draw label and score
            category = detection.categories[0]
            category_name = category.category_name
            category_name = '' if category_name is None else category_name
            probability = round(category.score, 2)
            result_text = category_name + ' (' + str(probability) + ')'
            text_location = (self.MARGIN + bbox.origin_x,
                             self.MARGIN + self.ROW_SIZE + bbox.origin_y)
            cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                        self.FONT_SIZE, self.TEXT_COLOR, self.FONT_THICKNESS)

        return annotated_image

    def model_set(self) -> vision.FaceDetector:
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceDetectorOptions(base_options=base_options,
                                             min_detection_confidence=0.6)
        detector = vision.FaceDetector.create_from_options(options)

        return detector

    def run(self):
        try:
            detector = self.model_set()
            while True:
                if not self.input_queue.empty():
                    msg, data = self.input_queue.get()
                    if msg == 0:
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=data)
                        detection_result = detector.detect(mp_image)
                        image_copy = np.copy(mp_image.numpy_view())
                        annotated_image = self.visualize(image_copy, detection_result)

                        self.output_queue.put((0, annotated_image))

                    elif msg == 99:
                        break

        except Exception as e:
            self.output_queue.put((99, str(e)))
            raise ValueError(e)


