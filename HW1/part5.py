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
#FIFTH VIDEO -- First/Face and Eye Detection
#               Second/Dynamic Background Substraction
#               Third/Motion Detection

    if sec > 0 and sec < 5: #FACE AND EYE DETECTION 
        temp = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        #faces = face_cascade.detectMultiScale(gray)

        for (x,y,w,h) in faces:
            cv2.rectangle(temp,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = temp[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                
        result = temp
        
    elif sec > 5 and sec < 10: # BACKGROUND SUBSTRACTION
        mask = subtractor.apply(frame)

        result = mask
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
    elif sec > 10 and sec < 33: #MOTION DETECTION
        
        ret, frame2 = cap.read()
        frame2=cv2.resize(frame2,(640,360),interpolation=cv2.INTER_AREA)
        
        if not ret:
            break
        
        # image difference
        img1 = cv2.absdiff(frame,frame2)
        
        # get threshold image
        gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(21,21),0)
        rett,thresh = cv2.threshold(blur,200,255,cv2.THRESH_OTSU)
        
        # combine frame and the image difference
        img2 = cv2.addWeighted(frame,0.9,img1,0.1,0)
        
        # get contours and set bounding box from contours
        img3, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,
                                                     cv2.CHAIN_APPROX_NONE)
        
        if len(contours) != 0:
            for c in contours:
                rect = cv2.boundingRect(c)
                height, width = img3.shape[:2]            
                if rect[2] > 0.2*height and rect[2] < 0.7*height and rect[3] > 0.2*width and rect[3] < 0.7*width: 
                    x,y,w,h = cv2.boundingRect(c)   # get bounding box of largest contour
                    img4=cv2.drawContours(img2, c, -1, (255,0,0), 2)
                    img5 = cv2.rectangle(img2,(x,y),(x+w,y+h),(0,0,255),2)# draw red bounding box in img
                else:
                    img5=img2
        else:
            img5=img2
            
        result = img2
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