#!/usr/bin/env python3

import argparse
import threading
import cv2
import numpy as np

COLOR_WHITE = (255, 255, 255)
COLOR_PURPLE = (147, 20, 255)

frame = None
pips = []
count = 0

frame_lock = threading.Lock()
pips_lock = threading.Lock()
count_lock = threading.Lock()


def naive_worker():
    global frame, pips, count

    while True:
        if frame is None:
            continue

        frame_lock.acquire()
        working_frame = np.copy(frame)
        frame_lock.release()

        working_frame = cv2.cvtColor(working_frame,
                                     cv2.COLOR_BGR2GRAY)
        _, working_frame = cv2.threshold(working_frame, 125, 255,
                                         cv2.THRESH_BINARY)
        bw = np.zeros(working_frame.shape, dtype=np.uint8)

        _, contours, hierarchy = cv2.findContours(working_frame,
                                                  cv2.RETR_TREE,
                                                  cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None:
            for i, h in enumerate(hierarchy[0]):
                if h[3] != -1:
                    cv2.drawContours(bw, contours, i, 255, cv2.FILLED)
                    cv2.drawContours(orig, contours, i, COLOR_PURPLE, 3)

        _, contours, _ = cv2.findContours(bw, cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)

        count_lock.acquire()
        count = len(contours)
        count_lock.release()

        pips_lock.acquire()
        pips = contours[:]
        pips_lock.release()


def haar_worker():
    global frame, pips, count
    cascade = cv2.CascadeClassifier('dice-pip-counter-haar/out/cascade.xml')

    while True:
        if frame is None:
            continue

        frame_lock.acquire()
        working_frame = np.copy(frame)
        frame_lock.release()

        output = cascade.detectMultiScale(frame, 1.2, 10)

        count_lock.acquire()
        count = len(output)
        count_lock.release()

        pips_lock.acquire()
        pips = output[:]
        pips_lock.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('source', action='store', help='IP Camera URL')
    parser.add_argument('algorithm', action='store',
                        choices=['naive', 'haar'],
                        help='Choose pip counting algorithm')

    args = parser.parse_args()

    vcap = cv2.VideoCapture(args.source)

    if args.algorithm == 'naive':
        t = threading.Thread(target=naive_worker, daemon=True)
    elif args.algorithm == 'haar':
        t = threading.Thread(target=haar_worker, daemon=True)

    t.start()

    while cv2.waitKey(1) & 0xFF != ord('q'):
        _, frame = vcap.read()
        if frame is not None:
            orig = np.copy(frame)

            count_lock.acquire()
            cv2.putText(orig, f'{count} pips', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, COLOR_WHITE, 2,
                        cv2.LINE_AA)
            count_lock.release()

            pips_lock.acquire()
            if args.algorithm == 'naive':
                cv2.drawContours(orig, pips, -1, COLOR_PURPLE, 3)
            elif args.algorithm == 'haar':
                for x, y, w, h in pips:
                    cv2.rectangle(orig, (x, y), (x+w, y+h), COLOR_PURPLE, 2)
            pips_lock.release()

            cv2.imshow('Preview', orig)

    vcap.release()
    cv2.destroyAllWindows()
