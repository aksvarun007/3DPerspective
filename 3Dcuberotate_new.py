import numpy as np
import cv2
import video
from common import anorm2, draw_str
from time import clock
import time
import sys
import math, pygame
import thread
from operator import itemgetter
class Point3D:
    def __init__(self, x = 0, y = 0, z = 0):
        self.x, self.y, self.z = float(x), float(y), float(z)
 
    def rotateX(self, angle):
        """ Rotates the point around the X axis by the given angle in degrees. """
        rad = angle * math.pi / 180
        cosa = math.cos(rad)
        sina = math.sin(rad)
        y = self.y * cosa - self.z * sina
        z = self.y * sina + self.z * cosa
        return Point3D(self.x, y, z)
 
    def rotateY(self, angle):
        """ Rotates the point around the Y axis by the given angle in degrees. """
        rad = angle * math.pi / 180
        cosa = math.cos(rad)
        sina = math.sin(rad)
        z = self.z * cosa - self.x * sina
        x = self.z * sina + self.x * cosa
        return Point3D(x, self.y, z)
 
    def rotateZ(self, angle):
        """ Rotates the point around the Z axis by the given angle in degrees. """
        rad = angle * math.pi / 180
        cosa = math.cos(rad)
        sina = math.sin(rad)
        x = self.x * cosa - self.y * sina
        y = self.x * sina + self.y * cosa
        return Point3D(x, y, self.z)
 
    def project(self, win_width, win_height, fov, viewer_distance):
        """ Transforms this 3D point to 2D using a perspective projection. """
        factor = fov / (viewer_distance + self.z)
        x = self.x * factor + win_width / 2
        y = -self.y * factor + win_height / 2
        return Point3D(x, y, self.z)


try:
	face_cascade = cv2.CascadeClassifier(sys.argv[1])
	eye_cascade = cv2.CascadeClassifier(sys.argv[2])
except:
	face_cascade=cv2.CascadeClassifier('/home/akshay/Documents/OpenCV/opencv-2.4.9/data/haarcascades/haarcascade_frontalface_default.xml')
	

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 700,
                       qualityLevel = 0.01,
                       minDistance = 0,
                       blockSize = 7 )

track_len = 5
detect_interval = 3
tracks = []
cap = cv2.VideoCapture(0)
frame_idx = 0

frame_gray = np.array([])

win_width = 1280
win_height = 720
pygame.init()

#variable to track thread
running = True

max_area = 0
face_x = 0
face_y = 0
face_w = 0
face_h = 0

min_val = 20000
max_val = 100000
scale = 1
old_scale = 1


#exponential averaging
alpha = 0.5
cent_x = 0
cent_y = 0
old_cent_x = 0
old_cent_y = 0
imp_x = 0
imp_y = 0


screen = pygame.display.set_mode((win_width, win_height))
pygame.display.set_caption("Simulation of a rotating 3D Cube")
		
clock = pygame.time.Clock()
d = 1
vertices = [
    Point3D(-d,d,-d),
    Point3D(d,d,-d),
    Point3D(d,-d,-d),
    Point3D(-d,-d,-d),
    Point3D(-d,d,d),
    Point3D(d,d,d),
    Point3D(d,-d,d),
    Point3D(-d,-d,d)
]

# Define the vertices that compose each of the 6 faces. These numbers are
# indices to the vertices list defined above.
faces_cube= [(0,1,2,3),(1,5,6,2),(5,4,7,6),(4,0,3,7),(0,4,5,1),(3,2,6,7)]

# Define colors for each face
colors = [(255,0,255),(255,0,0),(0,255,0),(0,0,255),(0,255,255),(255,255,0)]

angle = 0





def draw(frame,x,y):
    global win_width,win_height
    #draw box around the specified point
    cv2.rectangle(frame,(x-(win_width/8),y-(win_height/8)),(x+(win_width/8),y+(win_height/8)),(255,255,255),3)
    #draw basic lines
    rows,cols,ch=frame.shape



    
    cv2.line(frame,(0,0),(x-(win_width/8),y-(win_height/8)),(255,255,255),3)
    cv2.line(frame,(win_width,0),(x+(win_width/8),y-(win_height/8)),(255,255,255),3)
    cv2.line(frame,(win_width,win_height),(x+(win_width/8),y+(win_height/8)),(255,255,255),3)
    cv2.line(frame,(0,win_height),(x-(win_width/8),y+(win_height/8)),(255,255,255),3)
    #draw grid
    #for i in range(1,4):
        #cv2.line(frame,(0,(win_height/4)*i),(x-(win_width/8),(30*i)+(y-(win_height/8))),(255,255,255),1)
        #cv2.line(frame,((win_height/3)*i,0),((40*i)+(x-(win_width/8)),y-(win_height/8)),(255,255,255),1)
        #cv2.line(frame,(win_width,win_height/4*i),(x+(win_width/8),(30*i)+(y-(win_height/8))),(255,255,255),1)
        #cv2.line(frame,((win_height/3)*i,win_width),((40*i)+(x-(win_width/8)),y+(win_height/8)),(255,255,255),1)

    for i in range(1,7):
        cv2.line(frame,(0,(win_height*i/6)), (x-(win_width/8),(win_height*i/24)+(y-(win_height/8))),(255,255,255),1)
        cv2.line(frame,((win_width*i/6),0), (x-(win_width/8)+(win_width*i/24),(y-(win_height/8))),(255,255,255),1) 
        cv2.line(frame,(win_width,(win_height*i/6)), (x+(win_width/8),(win_height*i/24)+(y-(win_height/8))),(255,255,255),1)
        cv2.line(frame,((win_width*i/6),win_height), (x-(win_width/8)+(win_width*i/24),(y+(win_height/8))),(255,255,255),1)
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

def rotate_cube(x,y):
	global max_area,min_val,max_val,old_scale,scale,alpha
	for event in pygame.event.get():
        	if event.type == pygame.QUIT:
            		pygame.quit()
            		sys.exit()
    	clock.tick(50)
    	screen.fill((0,32,0))

    	# It will hold transformed vertices.
    	t = []
	
		    
    	for v in vertices:
            # Rotate the point around X axis, then around Y axis, and finally around Z axis.
            r = v.rotateX((x-320)/5).rotateY((y-240)/5).rotateZ(90)
            # Transform the point from 3D to 2D

            if max_area>max_val:
                scale = 3
            elif max_area<min_val:
                scale = 10
            else:
                scale = max_area*7.0/(max_val-min_val)
                if scale>=7:
                    scale = scale-2*(scale-7)
                elif scale <7:
                    scale = scale+2*(7-scale)
            #print scale
            scale = (1-alpha)*old_scale + alpha*scale
            old_scale = scale
            #print scale
            p = r.project(screen.get_width(),screen.get_height(), 256, 4)
            # Put the point in the list of transformed vertices
            t.append(p)

    	# Calculate the average Z values of each face.
    	avg_z = []
    	i = 0
	
    	for f in faces_cube:
		
		z = (t[f[0]].z + t[f[1]].z + t[f[2]].z + t[f[3]].z) / 4.0
		
		avg_z.append([i,z])
		i = i + 1

        # Draw the faces using the Painter's algorithm:
        # Distant faces are drawn before the closer ones.
        for tmp in sorted(avg_z,key=itemgetter(1),reverse=True):
		face_index = tmp[0]
		
		f = faces_cube[face_index]
		
	        pointlist = [(t[f[0]].x, t[f[0]].y), (t[f[1]].x, t[f[1]].y),
		             (t[f[1]].x, t[f[1]].y), (t[f[2]].x, t[f[2]].y),
		             (t[f[2]].x, t[f[2]].y), (t[f[3]].x, t[f[3]].y),
		             (t[f[3]].x, t[f[3]].y), (t[f[0]].x, t[f[0]].y)]

		pygame.draw.polygon(screen,colors[face_index],pointlist)
		    
        pygame.display.flip()


def haar():
    
    #face_cascade=cv2.CascadeClassifier('/home/naman/OpenCV-2.4.3/data/haarcascades/haarcascade_frontalface_default.xml')
    global face_x, face_y, face_w,face_h,tracks,frame_gray,frame_idx,detect_interval,max_area,running,face_cascade

    face_x_local = 0
    face_y_local = 0
    face_w_local = 0
    face_h_local = 0
    max_area = 0
    tracks_local = []

    time.sleep(1)
    
    while running:
        

        mask = np.zeros_like(frame_gray)
        mask[:] = 255
        for x, y in [np.int32(tr[-1]) for tr in tracks]:
            cv2.circle(mask, (x, y), 5, 0, -1)

        #determine the region in which face is present
        try:
            faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
        except:
            print 'no face'
            continue
        
        max_area = 0
        face_x_local = 0
        face_y_local = 0
        face_w_local = 0
        face_h_local = 0

        for (x,y,w,h) in faces:
            #find the largest face
            area = w*h
            if area > max_area:
                max_area = area
                face_x_local = x
                face_y_local = y
                face_w_local = w
                face_h_local = h
        p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                if (x>face_x_local and x<face_x_local+face_w_local and y>face_y_local and y<face_y_local+face_h_local):
                    tracks_local.append([(x, y)])

        face_x = face_x_local
        face_y = face_y_local
        face_w = face_w_local
        face_h = face_h_local
        tracks = tracks_local
        
    thread.exit()

thread.start_new_thread(haar,())

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
	


	valid=[]
	temp=[]
	for i in range(len(p0r)):
		#print p0r[i][0][0]	
		temp=[p0r[i][0][0]>face_x,p0r[i][0][0]<face_x+face_w,p0r[i][0][1]>face_y,p0r[i][0][1]<face_y+face_h]
		valid.append(all(temp))

				


	good=np.array([])
	#valid=x>face_x and x<face_x+face_w and y>face_y and y<    face_y+face_h
        #good = d < 1
	good=np.logical_and(d<1,valid)        
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
            cent_x=int(sum_x/index)
            cent_y=int(sum_y/index)
	   

         
            
	    #exponential averaging
	    cent_x = int((1-alpha)*old_cent_x + alpha*cent_x)
	    cent_y = int((1-alpha)*old_cent_y + alpha*cent_y)

	    #for lateral inversion
	    imp_x=win_width-cent_x
	    imp_y=win_height-cent_y

	    old_cent_x = cent_x
	    old_cent_y = cent_y

	    
	    cv2.circle(vis, (cent_x,cent_y), 15, (255, 0, 0), -1)
            #draw the blue circle in the center of the face and also draw the 3d perspective image on the frame1
	    draw(frame1,imp_x,imp_y)
	    #panoramaread(frame1,640-int(sum_x/index), int(sum_y/index))
	    
	rotate_cube(imp_x,imp_y)
		
        tracks = new_tracks
        #cv2.polylines(vis, [np.int32(tr) for tr in tracks], False, (255, 255, 0))
        draw_str(vis, (20, 20), 'track count: %d' % len(tracks))
        
        
    frame_idx += 1
    prev_gray = frame_gray
    cv2.imshow('lk_track', vis)
    #print vis.shape
    #cv2.imshow('3d',draw(np.zeros((720,1280,3),np.uint8), imp_x,imp_y))
    #print max_area
    #cv2.imshow('win',frame1)

    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        break
running = False
cap.release()
cv2.destroyAllWindows()
