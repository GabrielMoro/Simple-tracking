import cv2 as cv
import numpy as np

a = 0.001

vid = cv.VideoCapture('vid/WalkByShop1front.mpg')

_, aux1 = vid.read()
BG = cv.cvtColor(aux1, cv.COLOR_BGR2GRAY)
avg = np.float32(BG)

krnl = np.ones((3,3),np.uint8)

while(vid.isOpened()):
    ret, frame = vid.read()
    if (ret == False) or (cv.waitKey(7) == 27):
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    cv.accumulateWeighted(gray, avg, a)
    out = cv.convertScaleAbs(avg)

    # Background removal
    diff = cv.absdiff(gray, out)
    _, diff = cv.threshold(diff,22,255,cv.THRESH_BINARY)
    
    # Morphology
    noBG = cv.erode(diff, krnl, iterations = 2)
    noBG = cv.dilate(noBG, krnl, iterations = 6)
    noBG = cv.erode(noBG, krnl, iterations = 5)
    noBG = cv.dilate(noBG, krnl, iterations = 14)

    contours, hs = cv.findContours(noBG, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for c in contours:
        rect = cv.boundingRect(c)
        x, y, w, h = rect
        if(w < 45 or w > 120 or h < 45):
            continue
        cv.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)   

    cv.imshow('Grayscale',gray)
    cv.imshow('Average',out)
    cv.imshow('No BG', noBG)


vid.release()
cv.destroyAllWindows()
