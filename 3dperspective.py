import cv2
import cv2.cv as cv
import numpy as np
height = 480
width = 640

#variables for inversion
face_x = 0
face_y = 0

count = 0

frame = np.zeros((height,width,3), np.uint8)	
def camshift_tracking(img1, img2, bb):
        hsv = cv2.cvtColor(img1, cv.CV_BGR2HSV)
        mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        x0, y0, w, h = bb
        x1 = x0 + w -1
        y1 = y0 + h -1
        hsv_roi = hsv[y0:y1, x0:x1]
        mask_roi = mask[y0:y1, x0:x1]
        hist = cv2.calcHist( [hsv_roi], [0], mask_roi, [16], [0, 180] )
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX);
        hist_flat = hist.reshape(-1)
        prob = cv2.calcBackProject([hsv,cv2.cvtColor(img2, cv.CV_BGR2HSV)], [0], hist_flat, [0, 180], 1)
        prob &= mask
        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
        new_ellipse, track_window = cv2.CamShift(prob, bb, term_crit)
        return track_window
def draw(frame,x,y):
        #draw box around the specified point
        cv2.rectangle(frame,(x-80,y-60),(x+80,y+60),(255,255,255),3)
        #draw basic lines
        cv2.line(frame,(0,0),(x-80,y-60),(255,255,255),3)
        cv2.line(frame,(639,0),(x+80,y-60),(255,255,255),3)
        cv2.line(frame,(639,479),(x+80,y+60),(255,255,255),3)
        cv2.line(frame,(0,479),(x-80,y+60),(255,255,255),3)
        #draw grid
        for i in range(1,4):
            cv2.line(frame,(0,120*i),(x-80,(30*i)+(y-60)),(255,255,255),1)
            cv2.line(frame,(160*i,0),((40*i)+(x-80),y-60),(255,255,255),1)
            cv2.line(frame,(639,120*i),(x+80,(30*i)+(y-60)),(255,255,255),1)
            cv2.line(frame,(160*i,479),((40*i)+(x-80),y+60),(255,255,255),1)

        return frame

cap = cv2.VideoCapture(0)
ret,img = cap.read()
face_cascade=cv2.CascadeClassifier('/home/akshay/Documents/OpenCV/opencv-2.4.9/data/haarcascades/haarcascade_frontalface_default.xml')
faces=face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=2, minSize=(30,30), flags = cv.CV_HAAR_SCALE_IMAGE)

bb=(faces.item(0),faces.item(1),faces.item(2),faces.item(3))
#bb=(x,y,w,h)
frame = draw(frame,320,240)
#bb = (125,125,200,100) # get bounding box from some method
while True:
	frame = np.zeros((height,width,3), np.uint8)
	ret,img1 = cap.read()
	#print count
	count = count +1

        
	
	#print x,y
        
                #draw bounding box on img1
		#print bb
	'''if count>=10:	
		faces=face_cascade.detectMultiScale(img1, scaleFactor=1.2, minNeighbors=2, minSize=(0,0), flags = cv.CV_HAAR_SCALE_IMAGE)


		bb=(faces.item(0),faces.item(1),faces.item(2),faces.item(3))
		count = 0
		#continue'''
	(x,y,w,h)=bb	
	bb = camshift_tracking(img1, img, bb)
	img = img1
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imshow("CAMShift",img1)

        face_x = 640 - (x+(w/2))
        face_y = y+(h/2)
	draw(frame,face_x,face_y)

	cv2.imshow('draw',frame)
	k=cv2.waitKey(10)
	if k==27:
		break
cv2.destroyAllWindows()
