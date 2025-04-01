import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

height = 450
weidth = 450
right_box = [(0, height), (0, 0), (weidth, 0), (weidth, height)]

left_box = [(1020 - weidth, height), (1020 - weidth, 0),
            (1020, 0), (1020, height)]

start_time = time.time()
operation_time = time.time()
left = 0
right = 0

cap = cv2.VideoCapture(0)

# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        frame = cv2.resize(frame, (1020, 500))

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract Landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get Coordinates
            pinky = [int(landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].x * 1020),
                     int(landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].y * 500)]
            # print(pinky[0], pinky[1])

            # Left to Right
            enter_in_left_box = cv2.pointPolygonTest(
                np.array(left_box, np.int32), ((pinky[0], pinky[1])), False)
            enter_in_right_box = cv2.pointPolygonTest(
                np.array(right_box, np.int32), ((pinky[0], pinky[1])), False)

            if time.time() - operation_time > 2:
                if enter_in_left_box >= 0:
                    # print('Enter in LEFT')
                    if right > left:
                        pyautogui.press('right')
                        operation_time = time.time()
                        print("right")
                        left = 0
                        right = 0
                    else:
                        left = 1
                        start_time = time.time()

                elif enter_in_right_box >= 0:
                    # print('Enter in RIGHT')
                    if left > right:
                        pyautogui.press('left')
                        operation_time = time.time()
                        print("left")
                        left = 0
                        right = 0
                    else:
                        right = 1
                        start_time = time.time()

            if time.time() - start_time > 3 and time.time() - start_time < 4:
                left = 0
                right = 0

        except:
            pass

        # Render detections
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 177, 66),
                                   thickness=2, circle_radius=2),      # Joint Color
            mp_drawing.DrawingSpec(color=(245, 66, 230),
                                   thickness=2, circle_radius=2)       # Connection Color
        )

        cv2.polylines(image, [np.array(left_box, np.int32)],
                      True, (255, 0, 0), 2)
        cv2.polylines(
            image, [np.array(right_box, np.int32)], True, (255, 0, 0), 2)

        cv2.imshow('Mediapip Feed', image)
        # print('hii')

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
