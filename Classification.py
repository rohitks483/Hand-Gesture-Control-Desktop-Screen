import cv2
import tensorflow
from tensorflow.python.keras.models import load_model
import numpy as np
import pandas as pd
from keyboard import Keyboard

model = load_model(
    r'C:\Users\rohit\Videos\Gesture Project\resnetmodel.hdf5')

vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)

labels = pd.read_csv(
    r'C:\Users\rohit\Videos\Gesture Project\jester-v1-labels.csv', header=None)

buffer = []
cls = []
predicte_value = 0
final_label = ""
i = 1
while (vid.isOpened()):
    ret, frame = vid.read()
    if ret:
        image = cv2.resize(frame, (96, 64))
        image = image/255.0
        buffer.append(image)
        if(i % 16 == 0):
            buffer = np.expand_dims(buffer, 0)
            predicted_value = np.argmax(model.predict(buffer))
            cls = labels.iloc[predicted_value]
            print(cls)
            print(type(cls))
            print(cls.iloc[0])

            if(predicted_value == 0):
                final_label = 'Swiping left'
                Keyboard.key(Keyboard.VK_MEDIA_NEXT_TRACK)
            elif (predicted_value == 1):
                final_label = 'Swiping right'
                Keyboard.key(Keyboard.VK_MEDIA_PREV_TRACK)
            elif (predicted_value == 2):
                final_label = "Swiping down"
                Keyboard.key(Keyboard.VK_VOLUME_DOWN)
            elif (predicted_value == 3):
                final_label = "Swiping up"
                Keyboard.key(Keyboard.VK_VOLUME_UP)
            elif (predicted_value == 4):
                final_label = "pushing hand away"
            elif (predicted_value == 5):
                final_label = "pulling hand in"
            elif (predicted_value == 6):
                final_label = "sliding two fingres left"
            elif (predicted_value == 7):
                final_label = "sliding two fingres right"
            elif (predicted_value == 8):
                final_label = "sliding two fingres down"
                Keyboard.key(Keyboard.VK_VOLUME_DOWN)
            elif (predicted_value == 9):
                final_label = "sliding two fingres up"
                Keyboard.key(Keyboard.VK_VOLUME_UP)
            elif (predicted_value == 10):
                final_label = "pushing two fingres away"
            elif (predicted_value == 11):
                final_label = "pulling two fingers in"
            elif (predicted_value == 12):
                final_label = "rolling hand forward"
            elif (predicted_value == 13):
                final_label = "rolling hand backward"
            elif (predicted_value == 14):
                final_label = "turning hand clockwise"
            elif (predicted_value == 15):
                final_label = "turning hand counterclockwise"
            elif (predicted_value == 16):
                final_label = "zooming in with full hand"
            elif (predicted_value == 17):
                final_label = "zooming out with full hand"
            elif (predicted_value == 18):
                final_label = "zooming in with two fingers"
            elif (predicted_value == 19):
                final_label = "zooming out with two fingers"
            elif (predicted_value == 20):
                final_label = "thumb up"
            elif (predicted_value == 21):
                final_label = "thumb down"
            elif (predicted_value == 22):
                final_label = "shaking hand"
                Keyboard.key(Keyboard.VK_VOLUME_MUTE)
            elif (predicted_value == 23):
                final_label = "stop sign"
                Keyboard.key(Keyboard.VK_MEDIA_PLAY_PAUSE)
            elif (predicted_value == 24):
                final_label = "drumming fingers"
                Keyboard.key(Keyboard.VK_MEDIA_PLAY_PAUSE)
            elif (predicted_value == 25):
                final_label = "no gesture"
            else:
                final_label = "doing other things"

            cv2.imshow('frame', frame)
            buffer = []
        i = i+1
        text = "activity: {}".format(final_label)
        cv2.putText(frame, text, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 5)
        cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()
