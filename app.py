import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import time
from calculate import calculate

# opening webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

# Create an HandLandmarker object
model_path = 'operations.task'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
    
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.GestureRecognizerOptions(base_options=base_options,
                                          num_hands=2)
recognizer = vision.GestureRecognizer.create_from_options(options)

# Finger tip landmark indices
FINGER_TIP_IDS = [4, 8, 12, 16, 20]

# Drawing the hand landmarks and the handedness (left or right)
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

# Drawing landmarks and getting data
def drawLlandmarks_AND_inputData(rgb_image, result):
    hand_landmarks_list = result.hand_landmarks
    annotated_image = np.copy(rgb_image)
    finger_counter = None
    presence_of_hands = False

    # Landmarks Drawing
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())
    
    # Flipping the image for view
    annotated_image = cv2.flip(annotated_image, 1)
      
    # Detecting the Input

    # Detecting for addition and multiplication
    hand_data = {"Right": {}, "Left": {}}
    for hand_landmarks, hand in zip(hand_landmarks_list, result.handedness):
        presence_of_hands = True
        hand_type = hand[0].category_name
        hand_data[hand_type]["tip"] = hand_landmarks[8]
        hand_data[hand_type]["pip"] = hand_landmarks[6]
        hand_data[hand_type]["dip"] = hand_landmarks[7]

    # Efficient gesture detection for addition and multiplication
    right = hand_data["Right"]
    left = hand_data["Left"]

    # Checking for addition and multiplication
    if all(k in right for k in ("tip", "pip", "dip")) and all(k in left for k in ("tip", "pip", "dip")):
        if right['tip'] is not None and left['tip'] is not None and right['pip'] is not None and left['pip'] is not None:
            
            if (((right['tip'].y + 0.052) < left['tip'].y) and
                abs(left['pip'].x - right["pip"].x) < 0.04):
                #return +
                return (annotated_image, "+")
                
            elif ((right['tip'].x > left['tip'].x) and
                (right['tip'].y > (left['tip'].y - 0.01))):
                # return *
                return (annotated_image, "*")

    # Checking for operations from the model
    for idx in range(len(hand_landmarks_list)):
        top_gesture = result.gestures[idx][0]
        if top_gesture.score > 0.77:
            value = top_gesture.category_name
            match value:
                case "0":
                    return (annotated_image, "0")
                case "clear":
                    return(annotated_image, "c")
                case "equal":
                    return(annotated_image, "=")
                case "minus":
                    return(annotated_image, "-")
                case "devide":
                    return(annotated_image, "/")

    # Detecting for numbers
    for hand_landmarks, hand in zip(hand_landmarks_list, result.handedness):

        # Thumb detection (adjust logic for left/right hand)
        thumb_tip = hand_landmarks[4]
        index_tip = hand_landmarks[8]
        thresh = 0.035
        if hand[0].category_name == "Right":
            # For Right hand
            if (thumb_tip.x - index_tip.x) > thresh :
                if finger_counter is not None:
                    finger_counter += 1
                else:
                    finger_counter = 1
        else: # For Left hand
            if (index_tip.x - thumb_tip.x) > thresh:
                if finger_counter is not None:
                    finger_counter += 1
                else:
                    finger_counter = 1

        # Other fingers detection
        for i in range(1, 5):
            if hand_landmarks[FINGER_TIP_IDS[i]].y < hand_landmarks[FINGER_TIP_IDS[i] - 2].y:
                if finger_counter is not None:
                    finger_counter += 1
                else:
                    finger_counter = 1

    return (annotated_image, str(finger_counter).lower() if presence_of_hands else None)


# Initializing required variables
user_input = ""
ans = ""
last_input = None
last_time = 0
waiting_period = 2  # seconds

# With Camera Opened
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Resize
    frame = cv2.resize(frame, (1280, 720))

    # Creating Mediapipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Detect hand landmarks from the input image
    recognition_result = recognizer.recognize(mp_image)

    # Process the classification result. In this case, visualize it
    annotated_image, input = drawLlandmarks_AND_inputData(mp_image.numpy_view(), recognition_result)

    if last_time == 0 and input is not None:
        last_time = time.time()
    elif ((time.time() - last_time) > waiting_period and last_time != 0) or input == "none":
        last_time = time.time()

        if input is None:
            last_input = None
            last_time = 0
        elif input == "=":

            if user_input != "":
                user_input = str(calculate(user_input))
            ans = ""
        elif input == "c":
            # Clearing everything
            user_input = ""
            ans = ""
        elif ((input in ["+", "*", "/"] and last_input in ["+", "*", "/"]) or 
              input in ["+", "*", "/"] and user_input == ""):
            # Avoiding multiple operators back to back and first char as this operators
            pass
        elif input != "none":
            user_input += input
            # Call result func and store in ans
            if user_input not in ("","-"):
                ans = str(calculate(user_input))
            last_input = input
            last_time = time.time()

    # Adding the answerbar
    h, w = annotated_image.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = FONT_SIZE
    thickness = FONT_THICKNESS
    padding = 12  # vertical + horizontal padding inside rect

    # measure texts
    (text1_w, text1_h), baseline1 = cv2.getTextSize(user_input, font, font_scale, thickness)
    (text2_w, text2_h), baseline2 = cv2.getTextSize(ans, font, font_scale, thickness)

    # compute rectangle height to fit both texts with padding
    rect_height = text1_h + text2_h + baseline1 + baseline2 + padding * 3
    rect_x1, rect_x2 = 0, w
    rect_y2 = h
    rect_y1 = h - rect_height

    # draw full-width white rectangle at bottom
    cv2.rectangle(
        annotated_image,
        (rect_x1, rect_y1),
        (rect_x2, rect_y2),
        (255, 255, 255),
        thickness=-1
    )

    # compute centered x positions for each text
    text1_x = padding
    text2_x = padding

    # compute y positions: first text near top of rectangle, second below it
    text1_y = rect_y1 + padding + text1_h      # baseline-aligned y for first line
    text2_y = text1_y + padding + text2_h      # second line below first

    # draw texts (black color)
    cv2.putText(annotated_image, user_input, (text1_x, text1_y),
                font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    cv2.putText(annotated_image, ans, (text2_x, text2_y),
                font, font_scale, (132, 132, 132), thickness, cv2.LINE_AA)


    cv2.imshow("Gesture Calculator (Calculate just using your hands)", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
    if cv2.getWindowProperty("Gesture Calculator (Calculate just using your hands)", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()