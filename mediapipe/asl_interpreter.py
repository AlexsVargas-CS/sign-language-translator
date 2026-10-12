import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np

def drawHand(frame, hand): 
    # define which landmarks are connected
    hand_landmark_connections = [
       (0, 1), (1, 2), (2, 3), (3, 4), # thumb
       (0, 5), (5, 6), (6, 7), (7, 8), # index
       (9, 10), (10, 11), (11, 12), # middle
       (13, 14), (14, 15), (15, 16), # ring
       (0, 17), (17, 18), (18, 19), (19, 20), # pinky
       (5, 9), (9, 13), (13, 17) # inter-fingers
    ]

    # gets frame's height and width
    height, width = frame.shape[:2]

    # convert each landmark to coordinates on the screen
    points = []
    for landmark in hand:
        points.append([int(landmark.x * width), int(landmark.y * height)])

    # draws a line between each landmark
    for point1, point2 in hand_landmark_connections:
        cv2.line(frame, points[point1], points[point2], (0, 255, 0), 2)
    
    # draws a circle at each landmark
    for point in points: 
        cv2.circle(frame, point, 5, (0, 0, 255), -1)

def normalizeLandmarks(hand): 
    # create a numpy array 3x3 table for each landmark storing it's (x, y, z) coordinates
    points_list = []
    for landmark in hand:
        points_list.append([landmark.x, landmark.y, landmark.z])

    points = np.array(points_list)

    # subtract the wrist coordinates (stored at index 0) from all other landmarks
    points = points - points[0]

    # make everything the same size by dividing everything by the distance between wrist and base of middle finger
    distance = np.linalg.norm(points[9])
    points = points / distance

    # convert into single dimension array w/ 63 points
    points = points.flatten()
    return points




def handDetection(): 
    # describe how we want the hand detector configured
    options = vision.HandLandmarkerOptions(
        # hand_landmarker.task is a file in this directory that contains the neural networks to find the hands and its landmarks
        base_options = mp_python.BaseOptions(model_asset_path = "hand_landmarker.task"),
        running_mode = vision.RunningMode.VIDEO,
        num_hands = 1
    )

    # loads the model
    landmarker = vision.HandLandmarker.create_from_options(options)
    
    # open connection to the webcam
    cap = cv2.VideoCapture(0)
    start = time.time()

    while True: 
        # frame is a numpy array of pixels
        success, frame = cap.read()
        if not success:
            break

        # flip the frame
        frame = cv2.flip(frame, 1)

        # send the frame to mediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # puts into MediaPipe object
        mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data=rgb)
        
        # Video mode needs to know when each frame happened in ms
        timestamp_ms = int((time.time() - start) * 1000)

        # runs neural networks on the frame
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # read the result
        if (result.hand_landmarks and len(result.hand_landmarks) > 0):
            hand = result.hand_landmarks[0]
            drawHand(frame, hand)
            normalized_vector = normalizeLandmarks(hand)

        cv2.imshow("Webcam", frame)
       
        key = cv2.waitKey(1) & 0xFF
        # terminate on 'q' key press
        if (key == ord("q")): 
            break

    # cleanup
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

if __name__ == "__main__": 
    handDetection()
