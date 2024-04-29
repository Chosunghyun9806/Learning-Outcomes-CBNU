import multiprocessing
import os
import math
import cv2
import numpy as np
import mediapipe as mp

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style())

  return annotated_image


class FaceDetector_Land(multiprocessing.Process):

    def __init__(self, input_queue: multiprocessing.Queue, output_queue: multiprocessing.Queue, model_path: str):
        super(FaceDetector_Land, self).__init__()

        self.input_queue = input_queue
        self.output_queue = output_queue
        self.model_path = model_path

    def model_set(self) -> vision.FaceDetector:
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                               output_face_blendshapes=True,
                                               output_facial_transformation_matrixes=True,
                                               num_faces=20,
                                               min_face_detection_confidence=0.4,
                                               min_tracking_confidence=0.4)
        detector = vision.FaceLandmarker.create_from_options(options)

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
                        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
                        print(len(detection_result.face_landmarks))

                        self.output_queue.put((0, annotated_image))

                    elif msg == 99:
                        break

        except Exception as e:
            self.output_queue.put((99, str(e)))
            raise ValueError(e)


