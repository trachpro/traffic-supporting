#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
from license_model import LicenseNumberDetector

warnings.filterwarnings('ignore')


def main(yolo, license_number_detector):
    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    # deep_sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True

    video_capture = cv2.VideoCapture('/home/tupm/Downloads/Videos/Video_Traffic.mov')

    ret, frame = video_capture.read()
    cv2.namedWindow("display")
    def draw_rectangle(event, x, y, flags, param):

        image = frame.copy()
        global pt1, pt2, topLeft_clicked, bottomRight_clicked

        if event == cv2.EVENT_LBUTTONDOWN:
            # get coordinates of left corner
            if not topLeft_clicked:
                pt1 = (x, y)
                topLeft_clicked = True
            # get coordinates of right corner
            elif not bottomRight_clicked:
                pt2 = (x, y)
                cv2.line(image, pt1, pt2, (0, 255, 0), 3)
                cv2.imshow('display', image)
                bottomRight_clicked = True

    cv2.imshow('display', frame)
    cv2.setMouseCallback('display', draw_rectangle)
    cv2.waitKey(0)
    if writeVideo_flag:
        # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('VIDEO_50.avi', fourcc, 15, (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1

    fps = 0.0
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            print('break-----------------------')
            break
        t1 = time.time()
        cv2.line(frame, pt1, pt2, (0, 255, 0), 3)
        # image = Image.fromarray(frame)
        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        boxs, classes = yolo.detect_image(image)
        license_boxs = [box for i, box in enumerate(boxs) if classes[i] == 0]
        license_traffic_light = [box for i, box in enumerate(boxs) if classes[i] == 1]
        boxs = [box for i, box in enumerate(boxs) if classes[i] != 1 and classes[i] != 0]

        license_images = [Image.fromarray(frame[y:y + h, x: x + w, :]) for x, y, w, h in license_boxs]
        # if len(license_images) != 0:
        #     print(license_number_detector.detect_image(license_images))
        # print("box_num",len(boxs))
        features = encoder(frame, boxs)

        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            color = (0, 255, 0) if bbox[3] > pt1[1] else (0, 0, 255)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, color, 2)

        # for det in detections:
        #     bbox = det.to_tlbr()
        #     cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

        cv2.imshow('display', frame)

        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index) + ' ')
            if len(boxs) != 0:
                for i in range(0, len(boxs)):
                    list_file.write(
                        str(boxs[i][0]) + ' ' + str(boxs[i][1]) + ' ' + str(boxs[i][2]) + ' ' + str(boxs[i][3]) + ' ')
            list_file.write('\n')

        fps = (fps + (1. / (time.time() - t1))) / 2
        print("fps= %f" % (fps))

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    pt1 = (0, 0)
    pt2 = (0, 0)
    topLeft_clicked = False
    bottomRight_clicked = False
    main(YOLO(), LicenseNumberDetector())
