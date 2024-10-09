import cv2
import time
import PoseModule as pm
# global
# common
running=True
cTime=0
pTime=0
# laptop camera - 0
cam=cv2.VideoCapture(0)
detector=pm.poseDetector()
def update():
    global cTime,pTime,running,detector
    _,img=cam.read()
    img=detector.findPose(img)
    lmsList=detector.findLms(img)
    
    if lmsList:
        print(lmsList[0])

    # fps
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    # rendering
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