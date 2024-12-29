import mediapipe as mp
import cv2

class poseDetector():
    def __init__(self,
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5):
        
        self.static_image_mode=static_image_mode,
        self.model_complexity=model_complexity,
        self.smooth_landmarks=smooth_landmarks,
        self.enable_segmentation=enable_segmentation,
        self.smooth_segmentation=smooth_segmentation,
        self.min_detection_confidence=min_detection_confidence,
        self.min_tracking_confidence=min_tracking_confidence

        self.mpPose=mp.solutions.pose
        self.pose=self.mpPose.Pose(
            # self.static_image_mode,
            # self.model_complexity,
            # self.smooth_landmarks,
            # self.enable_segmentation,
            # self.smooth_segmentation,
            # self.min_detection_confidence,
            # self.min_tracking_confidence
        )
        self.mpDraw=mp.solutions.drawing_utils
    
    def findPose(self,img,draw=True):
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.result=self.pose.process(imgRGB)

        detectPose=self.result.pose_landmarks
        if detectPose and draw:
            self.mpDraw.draw_landmarks(
                img,
                detectPose,
                self.mpPose.POSE_CONNECTIONS
            )
        return img
    
    def findLms(self,img,draw=True):
        highlights=[12,24,26,11,23,25]
        lmsList=[]
        detectPose=self.result.pose_landmarks
        if detectPose:
            for id,lm in enumerate(detectPose.landmark):
                h,w,c=img.shape
                # convert to pixel
                px,py=int(lm.x*w),int(lm.y*h)
                lmsList.append([id,px,py])

                if draw:
                    if id in highlights[:3]:
                        cv2.circle(img, (px,py), 6, (255,0,0), 2)
                        cv2.circle(img, (px,py), 4, (244,244,244), cv2.FILLED)
                    
                    if id in highlights[3:]:
                        cv2.circle(img, (px,py), 6, (0,255,0), 2)
                        cv2.circle(img, (px,py), 4, (244,244,244), cv2.FILLED)
        
        return lmsList