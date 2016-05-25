import cv2
import skfuzzy as fuzz
import numpy as np

def fuzzyseg(img, hue, sat):
    """
    fuzzyseg(hsv, boundaries) -> mask
    Segments an Image in HSV color space, returning three masks with each pixel
    corresponding to the fuzzy level of red, blue and yellow.

    0 <= pixel_value <= 255
    """

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    size = img.shape[0:2] + (np.size(hue,0)+1,)

    mask = np.empty(size)
    
    x_hue, x_sat, _ = cv2.split(hsv)

    i = 0
    for boundaries in hue:
        mask[:,:,i] = np.apply_along_axis(fuzz.trapmf, 0, x_hue, boundaries)

        i += 1

    
    mask[:, :, i] = np.apply_along_axis(fuzz.trapmf, 0, x_sat, sat)

    return mask