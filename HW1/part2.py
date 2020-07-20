# -- Imports --
import numpy as np
from matplotlib import pyplot as plt
import cv2

# -- VIDEO CAPTURE ZONE --

cap = cv2.VideoCapture('fp3.mp4') #NAME WILL CHANGE DEPENDING ON THE VIDEO WHICH WILL BE PROCESSED

fps = cap.get(cv2.CAP_PROP_FPS)     # GET THE FPS RATE OF THE VIDEO
frame_ct = 0

width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)    # GET WIDTH
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # GET HEIGHT

#print(width,height)

fourcc = cv2.VideoWriter_fourcc(*'mp4v') #VIDEO WRITER FOR MAC
out = cv2.VideoWriter('out5.mp4', fourcc, fps, ( int(640), int(360) ), isColor=True) #INITIALIZE THE WRITER

subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50,   #INITIALIZE THE BACKGROUND SUBSTRACTOR
                                                    detectShadows=True)

while( cap.isOpened() ):    # WHEN VIDEO IS BEING ANALYZED 
    ret, frame = cap.read()
    
    frame_ct += 1
    
    sec = frame_ct / fps    # SEC CALCULATION 
    
    if not ret:
        break
    
    frame=cv2.resize(frame,(640,360),interpolation=cv2.INTER_AREA)
    
    print("Frame:", frame_ct)
    print("Second: %.3f" %sec)
    
# --------------------- INSERT YOUR CODE HERE ---------------------------------
#SECOND VIDEO -- Sobel x,y and xy

    if sec > 0 and sec < 6:
        
        if sec > 0 and sec < 0.5:
            result = frame
            
        elif sec > 0.5 and sec < 1.5:
            sobelx = cv2.Sobel(frame, int(-1 / cv2.CV_64F), 1, 0, ksize = 1)
            mid = cv2.cvtColor(sobelx, cv2.COLOR_BGR2GRAY)
            
        elif sec > 1.5 and sec < 2.5:
            sobelx = cv2.Sobel(frame, int(-1 / cv2.CV_64F), 1, 0, ksize = 5)
            mid = cv2.cvtColor(sobelx, cv2.COLOR_BGR2GRAY)
            
        elif sec > 2.5 and sec < 3.5:
            sobely = cv2.Sobel(frame, int(-1 / cv2.CV_64F), 0, 1, ksize = 1)
            mid = cv2.cvtColor(sobely, cv2.COLOR_BGR2GRAY)
            
        elif sec > 3.5 and sec < 4.5:
            sobely = cv2.Sobel(frame, int(-1 / cv2.CV_64F), 0, 1, ksize = 5)
            mid = cv2.cvtColor(sobely, cv2.COLOR_BGR2GRAY)
            
        else:
            sobel_all = cv2.Sobel(frame, int(-1 / cv2.CV_64F), 1, 1, ksize = 1)
            mid = cv2.cvtColor(sobel_all, cv2.COLOR_BGR2GRAY)
            
        if sec > 0.5:
            ret, mask = cv2.threshold(mid, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
            mid2 = cv2.cvtColor(mid, cv2.COLOR_GRAY2BGR)
            mid3 = cv2.bilateralFilter(mid2,5,75,75)
        
            indices = np.where(mask == 0)
        
        if sec > 0.5 and sec < 2.5:
            mid3[indices[0], indices[1], :] = [0,0,255]
            result = mid3
        elif sec > 2.5 and sec < 4.5:
            mid3[indices[0], indices[1], :] = [0,255,0]
            result = mid3
        elif sec > 4.5:
            mid3[indices[0], indices[1], :] = [255,0,0]
            result = mid3
        
    else:
        result = frame


# -----------------------------------------------------------------------------            

    out.write(result)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    #cv2.imshow('mid2', mid2)
    #cv2.imshow('mid3', mid3)
    cv2.imshow('Result', result)     # SHOWING THE RESULTING FRAME
    cv2.imshow('Original', frame)    # SHOWING THE ORIGINAL FRAME
    

out.release()
cap.release()
cv2.destroyAllWindows()