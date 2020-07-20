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
    # FOURTH VIDEO -- Highlighting the object and then using the template matching
    if sec > 0 and sec < 8:
        
        temp = frame.copy()
        mid = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        template = cv2.imread('template.png')
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        template = cv2.resize(template,None,fx=0.5,fy=0.5, interpolation=cv2.INTER_AREA)
           
        if sec > 0 and sec < 2.5:
            mid = cv2.medianBlur(mid,3)
            circles = cv2.HoughCircles(mid, cv2.HOUGH_GRADIENT, 1.5, 20, 
                                   param1=80, param2=100, minRadius=10, maxRadius=80)
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles [0,:]:
                    cv2.rectangle(temp, (i[0]-i[2]-10, i[1]-i[2]-10), 
                                  (i[0]+i[2]+10, i[1]+i[2]+10), (255,0,0), 3)
            result = temp
            out.write(result)
        
            
        elif sec > 2.5 and sec < 8:
            
            
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
            #w, h = template.shape[::-1]
            temp = cv2.matchTemplate(temp, template, cv2.TM_CCOEFF_NORMED)
            #cv2.normalize( temp, temp, 0, 1, cv2.NORM_MINMAX, -1 )
            #temp = temp.astype(np.uint8)
            dim = (640,360)
            result = cv2.resize(temp, dim, interpolation=cv2.INTER_AREA)
            
            #print(temp.shape)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            
            plt.imshow(result,cmap="gray")
            #cv2.imwrite('need',temp)
            print(temp.shape, result.shape, frame.shape)
            
            #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(mid2)
            
            #top_left = max_loc
            #bottom_right = (top_left[0] + w, top_left[1] + h)
            result = 255*result
            result = result.astype(np.uint32)
            #result = result/(result.max()/255.0)

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

''' 
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
'''

''' 
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
'''
'''
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
        
'''
'''
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
'''