import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *

model = YOLO('yolov8s.pt')

# Setup webpage
# st.title("Video Capture with OpenCV")


# def putData(enter, inside, exit):


# data_secction = st.empty()
# frame_placeholder = st.empty()
# stop_button_pressed = st.button("Stop")

# FOR peoplecount1.mp4
# area1 = [(312, 388), (289, 390), (474, 469), (497, 462)]    # Inside area

# area2 = [(279, 392), (250, 397), (423, 477), (454, 469)]    # Outside area

# # FOR room_demo1.mp4
# area1 = [(95, 512), (157, 488), (390, 548), (390, 590)]    # Inside area

# area2 = [(165, 485), (170, 450), (390, 490), (390, 540)]    # Outside area

# # FOR room_demo2.mp4
area1 = [(180, 450), (229, 420), (395, 427), (423, 466)]    # Inside area

area2 = [(225, 413), (225, 380),  (390, 380), (390, 426)]    # Outside area


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# cap = cv2.VideoCapture('peoplecount1.mp4')  # For demo video
cap = cv2.VideoCapture('room_demo2.mp4')  # For demo video
# cap = cv2.VideoCapture(1)  # For webcam of laptop


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
# print(class_list)

count = 0

tracker = Tracker()

people_entering = {}
entering = set()

people_exiting = {}
exiting = set()


while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("The video capture has ended || Any other error with the camera source")

    count += 1
    if count % 2 != 0:
        continue
    x = 393
    y = 700
    # 393, 700 for 9:16
    # 700, 393 for 16:9
    # 1020, 500 for peoplecount1

    frame = cv2.resize(frame, (x, y))
    results = model.predict(frame)

    a = results[0].boxes.boxes
    px = pd.DataFrame(a).astype("float")

    list = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'person' in c:
            list.append([x1, y1, x2, y2])

    bbox_id = tracker.update(list)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox

        xt = x4
        yt = y4
        cv2.circle(frame, (xt, yt), 5, (0, 255, 255), -1)

        # People Entering
        # Enter the outside area
        results = cv2.pointPolygonTest(
            np.array(area2, np.int32), ((xt, yt)), False)

        if results >= 0:
            people_entering[id] = (xt, yt)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)

        if id in people_entering:
            # Enter into inside area
            resules1 = cv2.pointPolygonTest(
                np.array(area1, np.int32), ((xt, yt)), False)
            if resules1 >= 0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                cv2.circle(frame, (xt, yt), 5, (255, 0, 255), -1)
                cv2.putText(frame, str(id), (x3, y3),
                            cv2.FONT_HERSHEY_COMPLEX, (0.5), (255, 255, 255), 1)
                entering.add(id)

        # People Exiting
            # Enter the inside area
        results2 = cv2.pointPolygonTest(
            np.array(area1, np.int32), ((xt, yt)), False)

        if results2 >= 0:
            people_exiting[id] = (xt, xt)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)

        if id in people_exiting:
            # Enter into outside area
            resules3 = cv2.pointPolygonTest(
                np.array(area2, np.int32), ((x4, y4)), False)
            if resules3 >= 0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                cv2.circle(frame, (xt, yt), 5, (255, 0, 255), -1)
                cv2.putText(frame, str(id), (x3, y3),
                            cv2.FONT_HERSHEY_COMPLEX, (0.5), (255, 255, 255), 1)
                exiting.add(id)

    cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 0, 0), 2)
    cv2.putText(frame, str('1'), (380, 480),
                cv2.FONT_HERSHEY_COMPLEX, (0.5), (0, 100, 255), 1)

    cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 0, 0), 2)
    cv2.putText(frame, str('2'), (380, 405),
                cv2.FONT_HERSHEY_COMPLEX, (0.5), (0, 100, 255), 1)

    # Visulize the data on the frame
    # Creat a top rectangle
    cv2.rectangle(frame, (0, 0), (x, 40), (0, 0, 0), -1)
    # Put the data on that
    cv2.putText(frame, str("Enter: " + str(len(entering)) + " Exit: " + str(len(exiting)) + " Inside: " + str(len(entering) - len(exiting))),
                ((x//2 - 100), 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # length of the text is 200

    # print(people_entering)
    # print(entering)

    # Output on local machine
    cv2.imshow("RGB", frame)

    # Output on web server
    # frame_placeholder.image(frame, channels="BGR")

    # with data_secction.container():
    #     cols = st.columns(3)
    #     with cols[0]:
    #         st.text('Temp: 31 C')

    #     with cols[1]:
    #         st.text('hiii')

    #     with cols[2]:
    #         st.text("Entered: " + str(len(entering)))
    #         st.text("Inside: " + str(len(entering) - len(exiting)))
    #         st.text("Exited: " + str(len(exiting)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
