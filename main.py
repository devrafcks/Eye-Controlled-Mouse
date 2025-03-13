import cv2
import mediapipe as mp
import pyautogui

cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

screen_w, screen_h = pyautogui.size()
speed_factor = 2.0
prev_x, prev_y = 0, 0
smoothing_factor = 0.5

def smooth_move(x, y):
    global prev_x, prev_y
    x = int(prev_x + smoothing_factor * (x - prev_x))
    y = int(prev_y + smoothing_factor * (y - prev_y))
    prev_x, prev_y = x, y
    return x, y

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape
    
    if landmark_points:
        landmarks = landmark_points[0].landmark
        
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))
            
            if id == 1:
                screen_x = int(screen_w * landmark.x * speed_factor)
                screen_y = int(screen_h * landmark.y * speed_factor)
                screen_x = max(0, min(screen_x, screen_w - 1))
                screen_y = max(0, min(screen_y, screen_h - 1))
                screen_x, screen_y = smooth_move(screen_x, screen_y)
                pyautogui.moveTo(screen_x, screen_y, duration=0.05)

        left = [landmarks[145], landmarks[159]]
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))
        
        if (left[0].y - left[1].y) < 0.008:
            pyautogui.click()
            pyautogui.sleep(1)

    cv2.imshow('Eye Controlled Mouse', frame)
    cv2.waitKey(1)
