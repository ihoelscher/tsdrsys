import cv2
from matplotlib import pyplot as plt
import fuzzyseg as fseg
import numpy as np
import gammacorr as gc

#folder = '/home/ihoelscher/imagens-test/snow fall/'
name = 'PICT0067.JPG'
img = cv2.imread(name)

plt.figure()
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

#img = gc.adjust_gamma(img, gamma=2)

""" ETAPA DE FILTRAGEM """

filter = "bi"

# y, cr, cb = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB))
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# y2 = clahe.apply(y)
# cr2 = clahe.apply(cr)
# cb2 = clahe.apply(cb)
# img = cv2.cvtColor(cv2.merge((y2, cr, cb)), cv2.COLOR_YCR_CB2BGR)

img = cv2.bilateralFilter(img, 5, 75, 75)
# cv2.imshow("lol", img)

# img = cv2.blur(img, (5,5))
# img = cv2.GaussianBlur(img, (5,5), 2)
# img = cv2.medianBlur(img, 5)

#
""" ETAPA DE SEGMENTACAO """

reda = [0, 0, 8, 12]
redb = [164, 168, 180, 181]
blue = [90, 96, 112, 118]
yellow = [16, 22, 28, 34]
sat = [20, 80, 255, 256]
hue = reda, redb, blue, yellow

mask = fseg.fuzzyseg(img, (reda, redb, blue, yellow), sat)
mred = (mask[:,:,0]+mask[:,:,1])*mask[:,:,4]
myellow = mask[:,:,3]*mask[:,:,4]

plt.figure()
plt.imshow(mred, cmap='gray')

plt.figure()
plt.imshow(myellow, cmap='gray')

#mred = np.array(255*mred, dtype=np.uint8)
#mblue = mask[:,:,2]*mask[:,:,4]
#mblue = np.array(255*mblue, dtype=np.uint8)
#myellow = mask[:,:,3]*mask[:,:,4]
#myellow = np.array(255*myellow, dtype=np.uint8)
#
#cv2.imshow("red"+filter+name, mred)
#cv2.imshow("blue"+filter+name, mblue)
#cv2.imshow("yellow"+filter+name, myellow)



#
# r = np.array(255*(red1 + red2)*level,dtype=np.uint8)
# b = np.array(blue*level,dtype=np.uint8)
# y = np.array(yellow*level,dtype=np.uint8)
#
#
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# r_img, b_img, y_img = fseg.fuzzyseg(hsv)
#
# element = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
# r_img = cv2.dilate(r_img, element)
# r_img = cv2.erode(r_img, element)

# element = cv2.getStructuringElement(cv2.MORPH_CROSS,(1,1))
# r_img = cv2.erode(r_img, element)

# hsv = cv2.cvtColor(blr, cv2.COLOR_BGR2HSV)
# r_blr, b_blr, y_blr = fseg.fuzzyseg(hsv)
#
# hsv = cv2.cvtColor(gauss, cv2.COLOR_BGR2HSV)
# r_gauss, b_gauss, y_gauss = fseg.fuzzyseg(hsv)
#

# hsv = cv2.cvtColor(median, cv2.COLOR_BGR2HSV)
# r_median, b_median, y_median = fseg.fuzzyseg(hsv)
#
# hsv = cv2.cvtColor(bi, cv2.COLOR_BGR2HSV)
# r_bi, b_bi, y_bi = fseg.fuzzyseg(hsv)



# cv2.imshow('red-original', r_img)
# cv2.imshow('red-mean', r_blr)
# cv2.imshow('red-gaussian', r_gauss)
# cv2.imshow('red-median', r_median)
# cv2.imshow('red-bilateral', r_bi)
