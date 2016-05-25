# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 16:49:41 2016

@author: ihoelscher
"""

import sys
sys.path.insert(0, '/home/ihoelscher/Documentos/python/imagestest')

import cv2
import csegmt as cs
import fuzzyseg as fseg
import numpy as np
from matplotlib import pyplot as plt
import samplePolygon as sP
import cmath
import makeFDInvariant as makeInv

def centroid(cnt):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    
    return cx, cy

def getKey(item):
    return cv2.arcLength(item, closed=True)

def constructObj(contours, rIntersec, rSize):
    cnt = sorted(contours, key=getKey, reverse=True)
    nCnt = len(cnt)
    
    obj = cnt[0];
    fundRect = cv2.boundingRect(obj)
    
    ratios = np.zeros(nCnt, dtype=float)
    
    for i in range(1, nCnt):
        
        rect = cv2.boundingRect(cnt[i])
        sRect = rect[2]*rect[3]
        
        sIntersec = max(0, min(fundRect[0] + fundRect[2], rect[0] + rect[2]) - max(fundRect[0], rect[0])) \
                    * max(0, min(fundRect[1] + fundRect[3], rect[1]+rect[3]) - max(fundRect[1], rect[1]))        
        
        ratios[i] = sIntersec / float(sRect)
        
        if cv2.contourArea(cnt[i]) > rSize*cv2.contourArea(obj) and ratios[i] > rIntersec:
            #print i, obj.shape, np.shape(cnt[i])
            obj = np.append(obj, cnt[i], axis=0)
            fundRect = cv2.boundingRect(obj)
    
    return obj, ratios
    
def rotatePoints(points, center, angle):
    s, c = np.sin(angle), np.cos(angle)
    R = np.array([[c, -s], [s, c]]);
    
    return np.dot(points - center,R) + center


name = "52.png"
img = cv2.imread(name)

redinterval = [([0, 60, 30], [10, 255, 255]), ([170, 60, 30], [180, 255, 255])]
output, mred = cs.csegmt(img, redinterval)

#adaptive kernel
sk = np.min(mred.shape)/100
if sk % 2 == 0:
    sk += 1

skernel = (sk, sk)


#filtering
mred = cv2.GaussianBlur(mred, skernel, 0)

#closing
kernel = np.ones(skernel, np.uint8)
mred = cv2.erode(cv2.dilate(mred, kernel, iterations=2), kernel, iterations=1)

#obtem a borda
boundary = cv2.Canny(mred, 10, 250, L2gradient=True)

#obtem todos contornos
contours, _ = cv2.findContours(boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours = sorted(contours, key=getKey, reverse=True)

obj, ratios = constructObj(contours, 0.25, 0.6)

hull = cv2.convexHull(obj)


#hullresized = hull*2;

c = centroid(hull)
#c2 = centroid(hullresized)
#hull2 = cv2.convexHull(obj2)
#c2 = centroid(hull2)
#cpimg = img.copy()
#cv2.drawContours(cpimg, [hull], 0, (0,255,0), 2)
#plt.figure()
#plt.imshow(cv2.cvtColor(cpimg, cv2.COLOR_BGR2RGB))

v = np.reshape(hull, (len(hull), len(hull[0][0])))

#vresized = np.reshape(hullresized, (len(hullresized), len(hullresized[0][0])))

#cv2.drawContours(cpimg, [v], 0, (0,255,0), 2)
#plt.figure()
#plt.imshow(cv2.cvtColor(cpimg, cv2.COLOR_BGR2RGB))

#vresized = rotatePoints(vresized, c2, np.pi/2)

#v2 += c 
#cv2.drawContours(cpimg, [vresized], 0, (0,255,0), 2)
#plt.figure()
#plt.imshow(cv2.cvtColor(cpimg, cv2.COLOR_BGR2RGB))

g = sP.uniSample(v-c, 1024)
#g2 = sP.uniSample(vresized, 1024)

G = np.fft.fft(g)
#G2 = np.fft.fft(g2)


def makeComplexInvariant(G):
    #Scale invariance: normalizar por |G[0]|
    Gout = G[1:]/np.sum(G[1:])
    
    #Rotation invariance: subtrair a fase do primeiro coeficiente
    r, phi = cmath.polar(Gout[0])
    for u in range(1, len(Gout)):
        r_u, phi_u = cmath.polar(Gout[u])
        phi_u -= u*phi
        Gout[u] = cmath.rect(r_u, phi_u)
        
    return Gout

#Go = makeComplexInvariant(G)
#Go2 = makeComplexInvariant(G2)
#Go = makeInv.makeFDInvariant(G)
#Go2 = makeInv.makeFDInvariant(G2)

#Make magnituasdfde Invariant
Go9 = np.abs(G[1:]/np.sum(G[1:]))
#Go2 = np.abs(G2[1:]/np.sum(G2[1:]))

plt.figure()
for x in g:
    plt.polar([0,np.angle(x)],[0,np.abs(x)],marker='o',color='r')

#for x in g2:
#    plt.polar([0,np.angle(x)],[0,np.abs(x)],marker='o',color='b')

#
#Go2 = np.abs(G2[1:]/np.abs(G2[1]))

#plt.figure()
#plt.plot(np.log10(np.abs(Go[:20])))
#plt.plot(np.log10(np.abs(Go2[:20])))
#plt.plot(np.log10(np.abs(Gol[:20])))
#plt.plot(np.log10(np.abs(Gol2[:20])))
#plt.plot(np.log10(np.abs(Goq[:20])))
#plt.plot(np.log10(np.abs(Goq2[:20])))
#plt.plot(np.log10(np.abs(Goc[:20])))
#plt.plot(np.log10(np.abs(Goc2[:20])))    
#plt.plot(np.log10(np.abs(Gocl[:20])))
#plt.plot(np.log10(np.abs(Gocl2[:20])))    

##mred = cv2.erode(mred, kernel)
##boundary = mred - cv2.erode(mred, kernel)
#
#plt.figure()
#plt.imshow(boundary, cmap='gray')
#
#rows,cols = boundary.shape
#M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
#dst = cv2.warpAffine(boundary,M,(cols,rows))
#
##plt.figure()
##plt.imshow(mred, cmap='gray')
#
#hum = np.log(np.square(cv2.HuMoments(cv2.moments(boundary))))

# Set up the detector with default parameters.
#detector = cv2.SimpleBlobDetector()
# 
## Detect blobs.
#keypoints = detector.detect(mred)
# 
## Draw detected blobs as red circles.
## cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
#im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# 
## Show keypoints
#plt.imshow(cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2RGB))
#

#im_floodfill = mred.copy()
# 
## Mask used to flood filling.
## Notice the size needs to be 2 pixels than the image.
#h, w = mred.shape[:2]
#mask = np.zeros((h+2, w+2), np.uint8)
# 
## Floodfill from point (0, 0)
#cv2.floodFill(im_floodfill, mask, (0,0), 255);
# 
## Invert floodfilled image
#im_floodfill_inv = cv2.bitwise_not(im_floodfill)
# 
## Combine the two images to get the foreground.
#im_out = mred | im_floodfill_inv
# 
## Display images.

#plt.imshow(mred, cmap='gray')

#contours, hierarchy = cv2.findContours(mred,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

#cv2.drawContours(img,contours,0,(0,255,0),3)
#plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


#sift = cv2.SIFT()
#kp, des = sift.detectAndCompute(mred, None)
#
#img = cv2.drawKeypoints(mred,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv2.imwrite('sift_circle.jpg',img)