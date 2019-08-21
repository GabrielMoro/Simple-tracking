import cv2 as cv
import numpy as np

a = 0.05

vid = cv.VideoCapture('vid/WalkByShop1front.mpg')

_, aux1 = vid.read()
aux2 = cv.cvtColor(aux1, cv.COLOR_BGR2GRAY)
avg = np.float32(aux2)

while(vid.isOpened()):
    ret, frame = vid.read()
    if (ret == False) or (cv.waitKey(1) == 27):
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    cv.accumulateWeighted(gray, avg, a)
    out = cv.convertScaleAbs(avg)

    cv.imshow('Grayscale',gray)
    cv.imshow('Average',out)
    cv.imshow('Test', (gray - out))


vid.release()
cv.destroyAllWindows()
