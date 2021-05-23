import cv2
from tensorflow import keras
import numpy as np

data_shape = (28, 28)


def pre_processing(img):
    processed = cv2.resize(img, data_shape)
    processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    processed = cv2.equalizeHist(processed)
    processed = 255-processed
    processed = processed/255
    processed = np.array(processed)
    processed = processed.reshape(1, 28, 28, 1)
    return processed


model = keras.models.load_model('trained_model.h5')

cam = cv2.VideoCapture(0)
while True:
    ret_val, img = cam.read()
    #img = cv2.flip(img, 1)
    prediction = pre_processing(img)
    number = np.argmax(model.predict(prediction))
    precent = np.amax(model.predict(prediction))
    if precent*100 > 80:
        cv2.putText(img, str(number), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
        cv2.putText(img, str(precent*100)[:4]+'%', (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    cv2.imshow('cam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()