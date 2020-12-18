import cv2 as cv
import imutils
from imutils.contours import sort_contours
import numpy as np


def readCharacters(image):
    image=cv.resize(image, (200, 200))
    #transform to gray and blur it
    grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurr = cv.GaussianBlur(grayImage,(5,5), 0)

    #detect edges, sort left to right
    edges = cv.Canny(blurr, 30, 150)
    contours = cv.findContours(edges.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sort_contours(contours, method="left-to-right")[0]

    rects = []   #for storing bounding boxes
    images = []  #for croped images

    for c in contours:
         #compute bound of the box
        (x, y, w, h) = cv.boundingRect(c)
        rect=(x,y,w,h)



        # remove all boxes that are too big or too small
        if (w >= 9 and w <= 100 and h >= 5 and h <= 150):
            rects.append(rect)    #add bounding box to list of bounding boxes
            # extract the character and threshold it to make the character
            # appear as *white* (foreground) on a *black* background, then
            # grab the width and height of the thresholded image
            roi = grayImage[y:y + h, x:x + w]
            threshold = cv.threshold(roi, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
            (trH, trW) = threshold.shape

            if trW > trH:
                thresh = imutils.resize(threshold, width=100)

    # otherwise, resize along the height
            else:
                thresh = imutils.resize(threshold, height=100)

            (trH, trW) = thresh.shape
            dX = int(max(0, 32 - trW) / 2.0)
            dY = int(max(0, 32 - trH) / 2.0)

            cropped = cv.copyMakeBorder(thresh, top=dY, bottom=dY,
                                left=dX, right=dX, borderType=cv.BORDER_CONSTANT,
                                value=(0, 0, 0))
            cropped = cv.resize(cropped, (100, 100))
            cropped = np.array(cropped)
            cropped = np.expand_dims(cropped, axis = 2)
            cv.imshow("slika", cropped)
            cv.waitKey(0)
            images.append(cropped)

    return images, rects        # returns list of croped images and list of coordintates (x,y) and width and height


def showCharacters(image):   # displays cropped picture of each found character
    image, boxes = readCharacters(image)

    for i in image:
        cv.imshow("character", i)
        cv.waitKey(0)