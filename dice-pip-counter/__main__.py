import cv2
import numpy as np

COLOR_WHITE = (255,)*3
COLOR_PURPLE = (147, 20, 255)
CAM_URL = 'http://192.168.8.103:8080/videofeed'

vcap = cv2.VideoCapture(CAM_URL)

while cv2.waitKey(1) & 0xFF != ord('q'):
    _, frame = vcap.read()
    if frame is not None:
        orig = np.copy(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, frame = cv2.threshold(frame, 125, 255, cv2.THRESH_BINARY)
        bw = np.zeros(frame.shape, dtype=np.uint8)

        _, contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None:
            for i, h in enumerate(hierarchy[0]):
                if h[3] != -1:
                    cv2.drawContours(orig, contours, i, COLOR_PURPLE, 3)
                    cv2.drawContours(bw, contours, i, 255, cv2.FILLED)

        _, hierarchy, _ = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        count = len(hierarchy)
        cv2.putText(orig, f'{count} pips', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, COLOR_WHITE, 2, cv2.LINE_AA)
        cv2.imshow('Preview', orig)

vcap.release()
cv2.destroyAllWindows()
