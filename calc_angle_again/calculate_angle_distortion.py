import math
import numpy as np
import cv2

lowerIR = np.array([220,220,220])
upperIR = np.array([255,255,255])
angles = []
pixels = []

for i in range(0,91,5):
    image = cv2.imread(str(i)+'.jpg',0)
    image = cv2.resize(image, (1920, 1080)) 
    image = cv2.flip(image,0)
    mask = cv2.inRange(image, 220, 255)
    im2, contours, hierarchy = cv2.findContours(mask.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #image = cv2.drawContours(image, contours, -1, (0,0,255), -1)
    
    try:
        cnt = contours[0]
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        center = (int(x),int(y))
        pixels.append(center[0])
        
        radius = int(radius)
        mask = cv2.circle(mask,center,radius,(0,255,0),2)
    except:
        pass
    
##    cv2.imshow('Frame',mask)
##    
##    key = cv2.waitKey(1) & 0xFF
##    
##
##    if key == ord('q'):
##        break

print pixels
    

