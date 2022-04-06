#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Detects and highlights face features from video using MediaPipe.
Extracts a 2D high-contrast triangular face mesh and features as video.

Requires OpenCV (v2), MediaPipe, ImageIO (w/ ffmpeg), Matplotlib:
  pip install opencv-python mediapipe imageio-ffmpeg matplotlib

Note: installing opencv-python-headless also fixes some OpenCV bugs.

Derived from:
* https://google.github.io/mediapipe/solutions/face_mesh
* https://gist.github.com/keithweaver/4b16d3f05456171c1af1f1300ebd0f12
* https://stackoverflow.com/a/4809040/18372312
* https://stackoverflow.com/a/51272988/18372312

Authored by Eric Easthope

MIT License
Copyright (c) 2022
"""

import cv2
import mediapipe as mp
import imageio
import numpy as np
import matplotlib.pyplot as plt

# Parameters
# Increase VIDEO_OUT_WIDTH and VIDEO_OUT_HEIGHT for more padding
# Increase VIDEO_OUT_FPS for faster videos
VIDEO_OUT_WIDTH = 500
VIDEO_OUT_HEIGHT = 500
VIDEO_OUT_FPS = 30

VIDEO_IN_FILENAME = "test-subject.mov"
VIDEO_OUT_FILENAME = "test-subject-face-mesh.mp4"

SHOW_PROCESSING = False

# Init Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# Annotation config
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Read video with ffmpeg
# Get video frames as Numpy array
video = imageio.get_reader(VIDEO_IN_FILENAME, "ffmpeg")
frames = np.array([frame for frame in video.iter_data()], dtype=np.uint8)

# Get array that bounds all nonzero array elements
# For 2D arrays only
# Set color to True if array has a third axis
def trim_zeros(A, color=False):
    if color:
        B = np.argwhere(A[:, :, 0])
    else:
        B = np.argwhere(A[:, :])

    (ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1

    if color:
        return A[ystart:ystop, xstart:xstop, :]
    else:
        return A[ystart:ystop, xstart:xstop]


# Detect face features from a single frame
# Draw face mesh contours on blank frame
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
) as face_mesh:
    # Get first frame
    # Init blank frame
    frame = frames[0]
    blank_frame = np.zeros(frame.shape)

    # Convert RGB frame to BGR before processing
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Detect features
    results = face_mesh.process(frame)

    # Draw face mesh contours over frame
    for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=blank_frame,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
        )

# Get face mesh array from nonzero frame values
# Get array size (pixels)
bounded_face_mesh = trim_zeros(blank_frame, color=True)
w, h = bounded_face_mesh.shape[:2]
if w >= VIDEO_OUT_WIDTH or h >= VIDEO_OUT_HEIGHT:
    raise Exception("Face mesh is larger than VIDEO_OUT_WIDTH or VIDEO_OUT_HEIGHT.")

# ALTERNATIVE: Load, annotate, and output video with OpenCV VideoCapture
# Load video with OpenCV
cap = cv2.VideoCapture(VIDEO_IN_FILENAME)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define output video codec
# Create VideoWriter object
fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
writer = cv2.VideoWriter(
    VIDEO_OUT_FILENAME, fourcc, VIDEO_OUT_FPS, (VIDEO_OUT_WIDTH, VIDEO_OUT_HEIGHT)
)

# Detect face features from input video frames
# Draw face contours on blank frames
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as face_mesh:
    # Read video frames while input video is open
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        # Init blank frame
        blank_frame = 0 * frame.copy()

        # OPTIONAL: Mark frame as not writeable to pass by reference
        # Might improve performance
        frame.flags.writeable = False

        # Convert RGB frame to BGR before processing
        # Detect features
        # Make frame writeable
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = face_mesh.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Draw face mesh annotations over frame
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=blank_frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                )
                mp_drawing.draw_landmarks(
                    image=blank_frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                )
                mp_drawing.draw_landmarks(
                    image=blank_frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
                )

                # If SHOW_PROCESSING, draw and show annotations over videos frames
                if SHOW_PROCESSING:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                    )
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                    )
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
                    )

                    # Flip frame horizontally for selfie-view display
                    cv2.imshow("MediaPipe Face Annotation", cv2.flip(frame, 1))

        # Get face mesh array from nonzero frame values
        # Get array size (pixels)
        bounded_face_mesh = trim_zeros(blank_frame, color=True)
        w, h = bounded_face_mesh.shape[:2]

        # Get padding based on VIDEO_OUT_WIDTH and VIDEO_OUT_HEIGHT
        # Padding tries to center face mesh annotations
        top = int((VIDEO_OUT_HEIGHT - h) / 2)
        bottom = VIDEO_OUT_HEIGHT - top - h
        left = int((VIDEO_OUT_WIDTH - w) / 2)
        right = VIDEO_OUT_WIDTH - left - w

        # Pad face mesh array to match VIDEO_OUT_WIDTH and VIDEO_OUT_HEIGHT
        padding = ((left, right), (top, bottom), (0, 0))
        padded_frame = np.pad(bounded_face_mesh, padding, "constant", constant_values=0)

        # Write padded frame to output video with VideoWriter
        writer.write(padded_frame)

	# Quit processing loop if escaped
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Print if processing finishes
print("Success!")

# Release VideoCapture
# Release VideoWriter
# Destroy OpenCV window
cap.release()
writer.release()
cv2.destroyAllWindows()
