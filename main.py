import cv2 as cv
import numpy as np

a = 0.05

vid = cv.VideoCapture('vid/WalkByShop1front.mpg')

_, aux1 = vid.read()
aux2 = cv.cvtColor(aux1, cv.COLOR_BGR2GRAY)
avg = np.float32(aux2)

krnl = np.ones((5,5),np.uint8)

while(vid.isOpened()):
    ret, frame = vid.read()
    if (ret == False) or (cv.waitKey(17) == 27):
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    cv.accumulateWeighted(gray, avg, a)
    out = cv.convertScaleAbs(avg)

    diff = cv.absdiff(gray, out)
    _,diff = cv.threshold(diff,20,255,cv.THRESH_BINARY)
    dilate = cv.dilate(diff, krnl, iterations = 1)

    contours, _ = cv.findContours(dilate, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cv.drawContours(gray, contours, -1, (0, 255, 0), 3)
    
    cv.imshow('Grayscale',gray)
    cv.imshow('Average',out)
    cv.imshow('No BG', dilate)



vid.release()
cv.destroyAllWindows()
