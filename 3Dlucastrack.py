import numpy as np
import cv2
import video
from common import anorm2, draw_str
from time import clock
import sys
try:
	face_cascade = cv2.CascadeClassifier(sys.argv[1])
	eye_cascade = cv2.CascadeClassifier(sys.argv[2])
except:
	face_cascade=cv2.CascadeClassifier('/home/akshay/Documents/OpenCV/opencv-2.4.9/data/haarcascades/haarcascade_frontalface_default.xml')
	eye_cascade=cv2.CascadeClassifier('/home/akshay/Documents/OpenCV/opencv-2.4.9/data/haarcascades/haarcascade_eye.xml')


lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 700,
                       qualityLevel = 0.01,
                       minDistance = 0,
                       blockSize = 7 )

track_len = 5
detect_interval = 10
tracks = []
cap = cv2.VideoCapture(0)
frame_idx = 0


def draw(frame,x,y):
        #draw box around the specified point
        cv2.rectangle(frame,(x-80,y-60),(x+80,y+60),(255,255,255),3)
        #draw basic lines
	rows,cols,ch=frame.shape
        #print rows
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
def panoramaread(frame,x,y):
	img=cv2.imread('/home/akshay/Desktop/image_stitching/result.jpg')
	rows,cols,ch = img.shape

	pts1 = np.float32([[x,y],[368+x,120+y],[28+x,387+y],[389+x,390+y]])

	pts2 = np.float32([[0+x,0+y],[300+x,0+y],[0+x,300+y],[300+x,300+y]])

	M = cv2.getPerspectiveTransform(pts1,pts2)

	dst = cv2.warpPerspective(img,M,(300,300))

	cv2.imshow('win1',dst)
	
	return frame


while True:
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    vis = frame.copy()
    frame1= np.zeros((480,640,3), np.uint8)

    if len(tracks) > 0:
        img0, img1 = prev_gray, frame_gray
        p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
        p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
        d = abs(p0-p0r).reshape(-1, 2).max(-1)
        good = d < 1
        new_tracks = []

        sum_x = 0
        sum_y = 0
        index = 0
        
        for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            tr.append((x, y))
            if len(tr) > track_len:
                del tr[0]
            new_tracks.append(tr)
            cv2.circle(vis, (x, y), 4, (0, 255, 0), -1)
	    

            sum_x = sum_x + x
            sum_y = sum_y + y
            index = index + 1
        if index>0:
            cv2.circle(vis, (int(sum_x/index), int(sum_y/index)), 10, (255, 0, 0), -1)
            #draw the blue circle in the center of the face and also draw the 3d perspective image on the frame1
	    draw(frame1,640-int(sum_x/index), int(sum_y/index))
	    panoramaread(frame1,640-int(sum_x/index), int(sum_y/index))
        tracks = new_tracks
        cv2.polylines(vis, [np.int32(tr) for tr in tracks], False, (255, 255, 0))
        draw_str(vis, (20, 20), 'track count: %d' % len(tracks))

        

    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(frame_gray)
        mask[:] = 255
        for x, y in [np.int32(tr[-1]) for tr in tracks]:
            cv2.circle(mask, (x, y), 5, 0, -1)

        #determine the region in which face is present
        faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
        max_area = 0
        face_x = 0
        face_y = 0
        face_w = 0
        face_h = 0

        for (x,y,w,h) in faces:
            #find the largest face
            area = w*h
            if area > max_area:
                max_area = area
                face_x = x
                face_y = y
                face_w = w
                face_h = h
	#cv2.rectangle(vis,(face_x,face_y),(face_x+face_w,face_y+face_h),(255,0,0),2) 
        #tracks = []
        p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):

                #if (x>(face_x + int((30/100)*face_w)) and x<(face_x+face_w - int((30/100)*face_w)) and y>(face_y+ int((20/100)*face_h)) and y<(face_y+face_h - int((20/100)*face_h))):
                if (x>face_x and x<face_x+face_w and y>face_y and y<    face_y+face_h):
                    tracks.append([(x, y)])
        
    frame_idx += 1
    prev_gray = frame_gray
    cv2.imshow('lk_track', vis) 
    cv2.imshow('win',frame1)

    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        break

cap.release()
cv2.destroyAllWindows()
