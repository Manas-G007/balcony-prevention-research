import cv2
import time
import PoseModule as pm
import numpy as np

# Global variables
running = True
cTime = 0
pTime = 0
safe = True  # Indicates if the person is safe

# Sample video or use camera feed
sample_video = "video/sample.mp4"
cam = cv2.VideoCapture(sample_video) 
detector = pm.poseDetector()

def calculate_angle_numpy(pointA, pointB, pointC):
    
    vectorAB = np.array(pointA) - np.array(pointB)
    vectorCB = np.array(pointC) - np.array(pointB)
    
    dot_product = np.dot(vectorAB, vectorCB)
    norm_product = np.linalg.norm(vectorAB) * np.linalg.norm(vectorCB)
    
    if norm_product == 0:
        return 0.0
    
    cos_theta = np.clip(dot_product / norm_product, -1.0, 1.0)
    
    angle = np.degrees(np.arccos(cos_theta))
    return angle

def draw_angle(img, p1, p2, p3, angle, color,orientation):

    cv2.line(img, tuple(p1), tuple(p2), color, 2)
    cv2.line(img, tuple(p2), tuple(p3), color, 2)

    mid_point = (int((p1[0] + p3[0]) / 2), int((p1[1] + p3[1]) / 2))
    cv2.putText(img, f"{int(angle)}", mid_point, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    px,py=p2
    cv2.circle(img, (px,py), int(angle//10), (0,255,0) if orientation else (255,0,0), 2)
    
def detect_climbing(img,lmsList):
    global safe
    # 12 24 26 -> left side angle
    p1 = [lmsList[12][1], lmsList[12][2]]
    p2 = [lmsList[24][1], lmsList[24][2]]
    p3 = [lmsList[26][1], lmsList[26][2]]
    left_angle = calculate_angle_numpy(p1, p2, p3)
    
    # Draw left angle
    draw_angle(img, p1, p2, p3, left_angle, (255,0,0),True)

    # 11 23 25 -> right side angle
    p1 = [lmsList[11][1], lmsList[11][2]]
    p2 = [lmsList[23][1], lmsList[23][2]]
    p3 = [lmsList[25][1], lmsList[25][2]]
    right_angle = calculate_angle_numpy(p1, p2, p3)
    
    # Draw right angle
    draw_angle(img, p1, p2, p3, right_angle, (0,255,0),False)
    
    if right_angle <= 70 or left_angle <= 70:
        if lmsList[23][2]<lmsList[25][2] or lmsList[24][2]<lmsList[26][2]:
            safe = False
    else:
        safe = True

def update():
    global cTime, pTime, running, detector, safe
    _, img = cam.read()
    
    if img is None:
        return
    
    img = detector.findPose(img,draw=False)
    lmsList = detector.findLms(img)
    
    # Detect climbing or proximity to the balcony edge
    if lmsList:
        detect_climbing(img,lmsList)

    # Calculate FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Resize image for better display
    width, height = 600, 400
    img = cv2.resize(img, (width, height))

    # Display text (Safe or Unsafe)
    cv2.putText(img, "Safe" if safe else "Unsafe", (10, 60),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 200, 0) if safe else (0, 0, 200), 2)

    # Display FPS
    cv2.putText(img, str(int(fps)), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 20), 1)
    
    color = (0, 200, 0) if safe else (0, 0, 200)
    thickness = 10
    cv2.rectangle(img, (0,0), (width,height), color, thickness)

    # Show the output image
    cv2.imshow("Balcony Preventor", img)
    cv2.waitKey(1)

def main():
    while running:
        update()

if __name__ == "__main__":
    main()