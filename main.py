import cv2
import time
import PoseModule as pm
import numpy as np

# global
# common
running=True
cTime=0
pTime=0
# laptop camera - 0

sample_video="video/sample.mp4"
cam=cv2.VideoCapture(sample_video)
detector=pm.poseDetector()

# util func
def findAngle(positions) -> int:
    _,x1,y1=positions[0]
    _,x2,y2=positions[1]
    _,x3,y3=positions[2]

    vector_21 = np.array([x1 - x2, y1 - y2])
    vector_23 = np.array([x3 - x2, y3 - y2])
    
    # Dot product and magnitudes of the vectors
    dot_product = np.dot(vector_21, vector_23)
    magnitude_21 = np.linalg.norm(vector_21)
    magnitude_23 = np.linalg.norm(vector_23)
    
    # Cosine of the angle
    cos_theta = dot_product / (magnitude_21 * magnitude_23)
    
    # Angle in radians
    angle_rad = np.arccos(cos_theta)
    
    # Convert angle to degrees
    angle_deg = np.degrees(angle_rad)

    centeroid=(x1+x2+x3)//3 ,(y1+y2+y3)//3 
    
    return centeroid,angle_deg

def update():
    global cTime,pTime,running,detector
    _,img=cam.read()
    img=detector.findPose(img)
    lmsList=detector.findLms(img)
    
    safe=True
    if lmsList:
        centeroid,angle=findAngle([lmsList[12],lmsList[24],lmsList[26]])
        if angle:
            safe=angle>130 
            x,y=centeroid
            cv2.putText(img,str(int(angle)),(x,y),
                cv2.FONT_HERSHEY_PLAIN,
                2,(0,255,0),1)

    # fps
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    width = 450
    height = 600
    img = cv2.resize(img, (width, height))

    # rendering
    cv2.putText(img,"safe" if safe else "not safe",(10,60),
        cv2.FONT_HERSHEY_COMPLEX,
        1,(0,200,0) if safe else (0,0,200),1)
    cv2.putText(img,str(int(fps)),(10,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,(20,20,20),1)
    cv2.imshow("Balcony Preventor",img)
    cv2.waitKey(1)

def main():
    while running:
        update()

if __name__=="__main__":
    main()