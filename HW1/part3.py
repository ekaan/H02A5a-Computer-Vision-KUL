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
#THIRD VIDEO -- 5 different kinds of Hough Circle function calls with different parameters
    if sec > 0 and sec < 11:
        
        temp = frame.copy()
        mid = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if sec > 0 and sec < 2: # DISCUSS ABOUT THE PARAMETERS IN THE SUBTITLES
            
            mid = cv2.bilateralFilter(mid, 5, 75, 75)
            circles = cv2.HoughCircles(mid, cv2.HOUGH_GRADIENT, 1.2, 3 2, 
                                       param1=50, param2=100, minRadius=0, maxRadius=100)
        
        elif sec > 8 and sec < 11: # DISCUSS ABOUT THE PARAMETERS IN THE SUBTITLES
            
            mid = cv2.bilateralFilter(mid, 5, 75, 75)
            circles = cv2.HoughCircles(mid, cv2.HOUGH_GRADIENT, 1.2, 20, 
                                       param1=50, param2=100, minRadius=0, maxRadius=100)
        
        elif sec > 4 and sec < 6: # DISCUSS ABOUT THE PARAMETERS IN THE SUBTITLES
            
            mid = cv2.medianBlur(mid,3)
            circles = cv2.HoughCircles(mid, cv2.HOUGH_GRADIENT, 1.5, 32, 
                                       param1=50, param2=100, minRadius=0, maxRadius=100)
        
        elif sec > 2 and sec < 4: # DISCUSS ABOUT THE PARAMETERS IN THE SUBTITLES
            
            mid = cv2.medianBlur(mid,5)
            circles = cv2.HoughCircles(mid, cv2.HOUGH_GRADIENT, 2, 32, 
                                       param1=50, param2=100, minRadius=0, maxRadius=100)
        
        
        elif sec > 6 and sec < 8: # DISCUSS ABOUT THE PARAMETERS IN THE SUBTITLES
            
            
            mid = cv2.medianBlur(mid,5)
            circles = cv2.HoughCircles(mid, cv2.HOUGH_GRADIENT, 1.5, 20, 
                                       param1=50, param2=100, minRadius=0, maxRadius=100)
    
            
            #try 1.2 for depth
        if circles is not None:
            circles = np.uint16(np.around(circles))
        
            for i in circles [0,:]:
                cv2.circle(temp,(i[0],i[1]),i[2],(0,255,0),2) # OUTER CIRCLES
                #cv2.circle(temp,(i[0],i[1]),2,(0,0,255),3) # INNER CIRCLES
            
        result = temp
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