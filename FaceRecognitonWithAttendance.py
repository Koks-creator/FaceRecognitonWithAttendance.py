from typing import List, Tuple
import os
from time import time
from datetime import datetime
import cv2
import numpy as np
import face_recognition
import csv

from db_tool import DBtool

db = DBtool(db_path="faces.db")
data = db.fetch_data()

images = []
class_names = []
faces_info = {}
dates = {}

for row in data:
    class_names.append(row['Name'])
    faces_info[row['Name'].lower()] = [row['Id'], row['Name'], row['Age'], row['Position'], row['Sex'], 0]
    images_bytes = row['ImageBytes']
    decoded = cv2.imdecode(np.frombuffer(images_bytes, np.uint8), -1)

    images.append(decoded)


if os.path.exists("attendance.csv") is False:
    with open("attendance.csv", "w", encoding="utf-8") as f:
        csv_writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["Name", "Position", "Date"])


def mark_attendance(name: str, position: str, date: datetime):
    str_date = date.strftime("%d-%m-%Y %H:%M:%S")

    with open("attendance.csv", "a", encoding="utf-8", newline="") as f:
        csv_writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([name, position, str_date])


def find_encodings(imgs: list) -> List[tuple]:
    encoded_imgs = []
    for img in imgs:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encoded_imgs.append(encode)

    return encoded_imgs


def get_text_center(text: str, x1: int, y1: int, x2: int, y2: int, size: float, t) -> list:
    # get boundary of this text
    textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, size, t)[0]

    w = x2 - x1
    h = y2 - y1

    textX = x1 + w // 2
    textY = y1 + h // 2

    return [textX, textY, textsize]


def fancy_bbox(img: np.array, x1: int, y1: int, x2: int, y2: int, color: Tuple[int, int, int], t: int) -> np.array:
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)

    line_w = min(int(w * 0.3), int(h * 0.3))

    cv2.line(img, (x1, y1), (x1, y1 + line_w), color, t)
    cv2.line(img, (x1, y1), (x1 + line_w, y1), color, t)

    cv2.line(img, (x1, (y1 + h)), (x1, (y1 + h) - line_w), color, t)
    cv2.line(img, (x1, (y1 + h)), (x1 + line_w, (y1 + h)), color, t)

    cv2.line(img, (x2, y1), (x2 - line_w, y1), color, t)
    cv2.line(img, (x2, y1), (x2, y1 + line_w), color, t)

    cv2.line(img, (x2, y2), (x2 - line_w, y2), color, t)
    cv2.line(img, (x2, y2), (x2, y2 - line_w), color, t)

    return img


known_faces_encodes = find_encodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(r"test.wmv")

final_img = False
p_time = 0
while True:
    detections = 0

    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = cv2.resize(img, (1280, 720))
    overlay = img.copy()

    img_s = cv2.resize(img, (0, 0), False, .25, .25)
    img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)

    faces_loc = face_recognition.face_locations(img_s)
    faces_encode = face_recognition.face_encodings(img_s, faces_loc)

    for encode_face, face_loc in zip(faces_encode, faces_loc):
        matches = face_recognition.compare_faces(known_faces_encodes, encode_face)
        face_dist = face_recognition.face_distance(known_faces_encodes, encode_face)

        match_index = np.argmin(face_dist)
        if matches[match_index]:
            class_name = class_names[match_index]
            user_id, name, age, position, sex, date = faces_info[class_name.lower()]
            y1, x2, y2, x1 = face_loc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

            # Transparency
            alpha = 0.4
            cv2.rectangle(overlay, (x1 - 5, y2 + 5), (x2 + 5, y2 + 90), (0, 0, 0), -1)
            img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

            img = fancy_bbox(img, x1, y1, x2, y2, (255, 255, 255), 6)
            text_x, text_y, text_size = get_text_center(f"Name: {name}", x1,  y2, x2, y2 + 30, .7, 2)
            cv2.putText(img, f"Name: {name}", (x1, (text_y + text_size[1] // 2) + 10),
                        cv2.FONT_HERSHEY_COMPLEX, .7, (255, 255, 255), 2)

            text_x2, text_y2, text_size2 = get_text_center(f"Pos: {position}", x1,  y2 + 30, x2, y2 + 60, .7, 2)
            cv2.putText(img, f"Pos: {position}", (x1, (text_y2 + text_size2[1] // 2) + 10),
                        cv2.FONT_HERSHEY_COMPLEX, .7, (255, 255, 255), 2)

            text_x3, text_y3, text_size3 = get_text_center(f"Age: {age}", x1,  y2 + 60, x2, y2 + 90, .7, 2)
            cv2.putText(img, f"Age: {age}", (x1, (text_y3 + text_size3[1] // 2) + 5),
                        cv2.FONT_HERSHEY_COMPLEX, .7, (255, 255, 255), 2)

            if faces_info[class_name.lower()][5] == 0:
                faces_info[class_name.lower()][5] = datetime.now()
                mark_attendance(name, position, faces_info[class_name.lower()][5])

            else:
                if (datetime.now() - faces_info[class_name.lower()][5]).seconds == 60:
                    faces_info[class_name.lower()][5] = datetime.now()
                    mark_attendance(name, position, faces_info[class_name.lower()][5])

            detections += 1

    c_time = time()
    fps = int(1 / (c_time - p_time))
    p_time = c_time
    cv2.putText(img, f"FPS: {fps}", (20, 40), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 200, 200), 2)
    cv2.putText(img, f"Detections: {detections}", (20, 65), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 200, 200), 2)

    cv2.imshow("Res", img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
