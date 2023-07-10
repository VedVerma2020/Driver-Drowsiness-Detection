import cv2
import numpy as np import
active = 0 status = "" color = (0, 0, 0)
def compute(ptA, ptB):


dist = np.linalg.norm(ptA - ptB) return dist
def blinked(a, b, c, d, e, f):


up = compute(b, d) + compute(c, e) down = compute(a, f)
ratio = up / (2.0 * down)

# Checking if it is blinked
if ratio > 0.25:
    return 2
elif (ratio > 0.21) and (ratio <= 0.25):
    return 1
else:
return 0


while True:
_, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

faces = detector(gray)
# detected face in faces array
for face in faces:
    x1 = face.left() y1 = face.top() x2 = face.right()
y2 = face.bottom()

face_frame = frame.copy() cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

landmarks = predictor(gray, face)
for n in range(0, 68):
x = landmarks.part(n).x y = landmarks.part(n).y

# Setting the radius of circles=2 pixels, colur=white
cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)

landmarks = face_utils.shape_to_np(landmarks)

left_blink = blinked(landmarks[36], landmarks[37], landmarks[38],
                     landmarks[41], landmarks[40], landmarks[39])
right_blink = blinked(landmarks[42], landmarks[43], landmarks[44],
                      landmarks[47], landmarks[46], landmarks[45])

if (left_blink == 0 or right_blink == 0):
    sleep += 1
drowsy = 0
active = 0
if (sleep > 6):
status = "SLEEPING !!!" color = (255, 0, 0) sound2.play() time.sleep(1) sound2.stop()

elif (left_blink == 1) or (right_blink == 1):


sleep = 0
active = 0
drowsy += 1
if (drowsy > 6):
status = "Drowsy !" color = (0, 0, 255) sound1.play()
