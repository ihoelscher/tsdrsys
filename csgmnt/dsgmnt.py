import numpy as np
import cv2 as cv

def csegmt(img, boundaries):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    size = img.shape[0:2] + (np.size(boundaries,0),)

    mask = np.uint8(np.ones(size))
    i = 0

    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        mask[:,:,i] = cv.inRange(hsv, lower, upper)
        i += 1

    outmask = np.uint8(mask.sum(axis=2))

    output = cv.bitwise_and(img, img, mask=outmask)

    return output, outmask