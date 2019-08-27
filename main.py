import cv2 as cv
import numpy as np

a = 0.001

vid = cv.VideoCapture('vid/WalkByShop1front.mpg')

_, aux1 = vid.read()
BG = cv.cvtColor(aux1, cv.COLOR_BGR2GRAY)
avg = np.float32(BG)
frame_width = int(vid.get(3))
frame_height = int(vid.get(4))

out = cv.VideoWriter('vid/output.avi', cv.VideoWriter_fourcc(
    'M', 'J', 'P', 'G'), 60, (frame_width, frame_height))

krnl = np.ones((3,3),np.uint8)

while(vid.isOpened()):
    ret, frame = vid.read()
    if (ret == True):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Background removal
        cv.accumulateWeighted(gray, avg, a)
        bg = cv.convertScaleAbs(avg)
        diff = cv.absdiff(gray, bg)
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
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        

        cv.imshow('Video',frame)
        out.write(frame)
        cv.imshow('Average',bg)
        cv.imshow('No BG', noBG)
        if (cv.waitKey(7) == 27):
            break
    else:
        break

    
out.release()
vid.release()
cv.destroyAllWindows()
