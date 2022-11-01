import cv2 as cv
from cv2 import aruco
import numpy as np
dictionary=aruco.Dictionary_get(aruco.DICT_4X4_50)
parameter=aruco.DetectorParameters_create()

capture=cv.VideoCapture(0)

while True:
    ret,frame=capture.read()
    if not ret:
        break
    gray_frame=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    marker_corners,marker_ID,reject=aruco.detectMarkers(gray_frame,dictionary,parameters=parameter)
    for corners in marker_corners:
        cv.polylines(frame, [corners.astype(np.int32)],True, (0,255,255),4,cv.LINE_AA)
        corners=corners.reshape(4,2)
        corners=corners.astype(int)
        top_right=corners[0].ravel()
        cv.putText(frame,f"id: {marker_ID[0]}",top_right,cv.FONT_HERSHEY_PLAIN,1.3,(0,255,0),2,cv.LINE_AA)

    
    cv.imshow("frame",frame)
    key=cv.waitKey(1)
    if key==ord("q"):
        break

capture.release()
cv.destroyAllWindows()
