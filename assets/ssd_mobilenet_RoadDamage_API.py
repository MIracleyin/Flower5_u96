import numpy as np

import re
import math
import random
import cv2
import time
import math


from rknn.api import RKNN


INPUT_SIZE = 300

NUM_RESULTS = 1917
NUM_CLASSES = 9

Y_SCALE = 10.0
X_SCALE = 10.0
H_SCALE = 5.0
W_SCALE = 5.0

MIN_SCORE = 0.3
NMS_THRESHOLD = 0.45

colorArray = [
    (139, 0, 0, 255),
    (139, 0, 139, 255),
    (0, 0, 139, 255),
    (0, 100, 0, 255),
    (139, 139, 0, 255),
    (209, 206, 0, 255),
    (0, 127, 255, 255),
    (139, 61, 72, 255),
    (0, 255, 0, 255)
]


def expit(x):
    return 1. / (1. + math.exp(-x))


def load_box_priors():
    box_priors_ = []
    fp = open('assets/box_priors.txt', 'r')
    ls = fp.readlines()
    for s in ls:
        aList = re.findall('([-+]?\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?', s)
        for ss in aList:
            aNum = float((ss[0]+ss[2]))
            box_priors_.append(aNum)
    fp.close()

    box_priors = np.array(box_priors_)
    box_priors = box_priors.reshape(4, NUM_RESULTS)

    return box_priors


def decodeCenterSizeBoxes(predictions, boxPriors):  # TODO:当前耗时约1.5秒，需要优化
    for i in range(0, NUM_RESULTS):
        ycenter = predictions[i * 4 + 0] / Y_SCALE * \
            boxPriors[2][i] + boxPriors[0][i]
        xcenter = predictions[i * 4 + 1] / X_SCALE * \
            boxPriors[3][i] + boxPriors[1][i]
        h = math.exp(predictions[i * 4 + 2] / H_SCALE) * boxPriors[2][i]
        w = math.exp(predictions[i * 4 + 3] / W_SCALE) * boxPriors[3][i]

        ymin = ycenter - h / 2.0
        xmin = xcenter - w / 2.0
        ymax = ycenter + h / 2.0
        xmax = xcenter + w / 2.0

        predictions[i * 4 + 0] = ymin
        predictions[i * 4 + 1] = xmin
        predictions[i * 4 + 2] = ymax
        predictions[i * 4 + 3] = xmax

    return predictions


def scaleToInputSize(outputClasses, output, numClasses):  # TODO:当前耗时约0.5秒，需要优化
    validCount = 0
    # Scale them back to the input size.
    for i in range(0, NUM_RESULTS):
        topClassScore = -1000.0
        topClassScoreIndex = -1
        # Skip the first catch-all class.
        for j in range(1, NUM_CLASSES):
            score = expit(outputClasses[i * numClasses + j])
            if score > topClassScore:
                topClassScoreIndex = j
                topClassScore = score

        if topClassScore >= MIN_SCORE:
            output[0][validCount] = i
            output[1][validCount] = topClassScoreIndex
            validCount = validCount + 1
    return validCount, output


def CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1):
    w = max(0.0, min(xmax0, xmax1) - max(xmin0, xmin1))
    h = max(0.0, min(ymax0, ymax1) - max(ymin0, ymin1))
    i = w * h
    u = (xmax0 - xmin0) * (ymax0 - ymin0) + \
        (xmax1 - xmin1) * (ymax1 - ymin1) - i
    if u <= 0.0:
        return 0.0
    else:
        return i/u


def nms(validCount,  outputLocations,   output):
    for i in range(0, validCount):
        if output[0][i] == -1:
            continue
        n = output[0][i]

        for j in range(i+1, validCount):
            m = output[0][j]
            if m == -1:
                continue
            xmin0 = outputLocations[n * 4 + 1]
            ymin0 = outputLocations[n * 4 + 0]
            xmax0 = outputLocations[n * 4 + 3]
            ymax0 = outputLocations[n * 4 + 2]

            xmin1 = outputLocations[m * 4 + 1]
            ymin1 = outputLocations[m * 4 + 0]
            xmax1 = outputLocations[m * 4 + 3]
            ymax1 = outputLocations[m * 4 + 2]

            iou = CalculateOverlap(xmin0, ymin0, xmax0,
                                   ymax0, xmin1, ymin1, xmax1, ymax1)

            if iou >= NMS_THRESHOLD:
                output[0][j] = -1
    return output


# Init RKNN
#
# Create RKNN object
rknn = RKNN()
# load labels and boxPriors
labels = ["D00", "D01", "D10", "D11", "D20", "D40", "D43", "D44"]
box_priors = load_box_priors()


def RDDA_init():
    # Direct Load RKNN Model
    print('--> Loading model...')
    rknn.load_rknn('assets/ssd_mobilenet_RoadDamageDetector.rknn')
    # init runtime environment
    print('--> Initializing runtime environment...')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Initialize runtime environment failed!')
        exit(ret)
    print('--> Running model')


def RDDA_detect(frame):
    time0 = time.time()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE),
                     interpolation=cv2.INTER_CUBIC)
    # Inference
    outputs = rknn.inference(inputs=[img])
    time1 = time.time()

    # Post Process
    predictions = outputs[1].flatten()
    outputClasses = outputs[0].flatten()

    # # transform

    # tt0 = time.time()
    predictions = decodeCenterSizeBoxes(predictions, box_priors)
    # tt1 = time.time()  # XXX: 约1.5s
    output = [[0 for y in range(NUM_RESULTS)] for x in range(2)]
    validCount, output = scaleToInputSize(
        outputClasses, output, NUM_CLASSES)
    # tt2 = time.time()  # XXX: 约0.5s
    # print("decodeCenterSizeBoxes: %4.2f ms, scaleToInputSize: %4.2f ms" % (
    #     1000*(tt1-tt0), 1000*(tt2-tt1)))
    # print("validCount: %d\n" % validCount)

    damageNum = 0

    if validCount < 100:
        output = nms(validCount, predictions, output)
        for i in range(0, validCount):
            if output[0][i] == -1:
                continue
            n = output[0][i]
            topClassScoreIndex = output[1][i] - 1  # FIXME: may have bug

            if topClassScoreIndex >= 6:
                continue

            xmin = predictions[n * 4 + 1] * frame.shape[1]
            ymin = predictions[n * 4 + 0] * frame.shape[0]
            xmax = predictions[n * 4 + 3] * frame.shape[1]
            ymax = predictions[n * 4 + 2] * frame.shape[0]
            
            print("%d %d"%(frame.shape[1],frame.shape[0]))
            if ymax > 320: # XXX:只判断中点位于画面下半部分的
                damageNum = damageNum + 1
                label = labels[topClassScoreIndex]

                print("%s @ (%d, %d) (%d, %d) " % (
                    label, xmin, ymin, xmax, ymax))
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                              colorArray[topClassScoreIndex % 9], 3)
                # cv2.putText(frame, label, (int(xmin), int(xmax) - 12),
                #             1, 2, (0, 255, 0))
    time2 = time.time()
    print("damageNum: %d\n" % damageNum)
    print("Inference Time: %4.2f ms, Post Process Time: %4.2f ms, Total: %4.2f ms" % (
        1000*(time1-time0), 1000*(time2-time1), 1000*(time2-time0)))
    return damageNum, 1000*(time2-time0), frame


def RDDA_release():
    rknn.release()


if __name__ == '__main__':
    RDDA_init()
    while True:
        num, cost, show = RDDA_detect()
        cv2.imshow('road damage detect', show)
        if cv2.waitKey(1) == ord('q'):
            RDDA_release()
            break
