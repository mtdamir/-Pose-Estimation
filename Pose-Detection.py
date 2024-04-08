import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


cap = cv2.VideoCapture(-1)
while cap.isOpened():
    _, frame = cap.read()
    try:
        # frame = cv2.resize(frame,(350, 600))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2RGB)

        pose_result = pose.process(frame_rgb)

        mp_drawing.draw_landmarks(frame, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow("Outpute",frame)
    except:
        break

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



