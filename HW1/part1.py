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
#FIRST VIDEO -- Filter, Blur and Grabbing

    if sec > 1 and sec < 2:
        
        result = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    elif sec > 2 and sec < 3:
        
        result = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
    elif sec > 3 and sec < 4:
        
        result = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
    elif sec > 4 and sec < 5:
        
        result = cv2.blur(frame,(5,5))
        
    elif sec > 5 and sec < 6:
        
        result = cv2.blur(frame,(10,10))
        
    elif sec > 6 and sec < 7:
        
        result = cv2.GaussianBlur(frame,(5,5),cv2.BORDER_DEFAULT)
        
    elif sec > 7 and sec < 8:
        
        result = cv2.GaussianBlur(frame,(15,15),cv2.BORDER_DEFAULT)
        
    elif sec > 8 and sec < 9:
        
        result = cv2.GaussianBlur(frame,(25,25),cv2.BORDER_DEFAULT)
        
    elif sec > 9 and sec < 10.5:
        
        result = cv2.bilateralFilter(frame,15,75,75) 
        # 2nd = depth
        # 3rd = sigma color
        # 4th = sigma space
        # It is easy to note that all these denoising filters smudge the 
        # edges, while Bilateral Filtering retains them.
        
    elif sec > 10.5 and sec < 12:
        
        result = cv2.bilateralFilter(frame,100,75,75) 
        
    elif sec > 12 and sec < 16: # DONT FORGET TO ADD THE THRESHOLDS 
        
        low_b = 30
        high_b = 170
        low_g = 40
        high_g = 160
        low_r = 127
        high_r = 255
        
        bgr_thr = cv2.inRange(frame, (low_b, low_g, low_r) , (high_b, high_g, high_r))
        
        result = cv2.cvtColor(bgr_thr, cv2.COLOR_GRAY2BGR)
        
    elif sec > 16 and sec < 22: # DONT FORGET TO ADD THE THRESHOLDS AGAIN BRUH
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        low_h = 100
        high_h = 200
        low_s = 30
        high_s = 255
        low_v = 0
        high_v = 255
        
        hsv_thr = cv2.inRange(hsv, (low_h, low_s, low_v), (high_h, high_s, high_v))
        
        result = cv2.cvtColor(hsv_thr, cv2.COLOR_GRAY2BGR)
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
