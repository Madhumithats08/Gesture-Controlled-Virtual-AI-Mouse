import cv2
import os
import mediapipe as mp
import pyautogui
import logging
from pynput.mouse import Button, Controller
from plyer import notification
import random
import util  # Ensure this module is implemented correctly
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
import math

# Initialize audio control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

# Set volume (range: 0.0 to 1.0)
def set_volume(level):
    volume.SetMasterVolumeLevelScalar(level, None)

# Example usage
set_volume(0.5)  # Set volume to 50%
# Get the current volume level
current_volume = volume.GetMasterVolumeLevelScalar()

import screen_brightness_control as sbc

try:
    # Get current brightness of the primary display
    current_brightness = sbc.get_brightness(display=0)[0]
    
    # Increase brightness by 10
    sbc.set_brightness(min(current_brightness + 10, 100))
    print(f"Brightness increased to {min(current_brightness + 10, 100)}")
except Exception as e:
    print(f"Error: {e}")



logging.basicConfig(filename='accuracy_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')
mouse = Controller()

# Screen dimensions
screen_width, screen_height = pyautogui.size()


# Mediapipe Hands setup
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

# Metrics for each gesture
metrics = {
    "left_click": {"tp": 0, "fp": 0, "fn": 0},
    "right_click": {"tp": 0, "fp": 0, "fn": 0},
    "double_click": {"tp": 0, "fp": 0, "fn": 0},
    "screenshot": {"tp": 0, "fp": 0, "fn": 0},

    "volume_up": {"tp": 0, "fp": 0, "fn": 0},
    "volume_down": {"tp": 0, "fp": 0, "fn": 0},
    "brightness_up": {"tp": 0, "fp": 0, "fn": 0},
    "brightness_down": {"tp": 0, "fp": 0, "fn": 0}
}

def update_metrics(gesture, detected):
    """Update metrics based on detected and expected gestures."""
    if detected:
        metrics[gesture]["tp"] += 1
    else:
        metrics[gesture]["fp"] += 1

# def log_metrics():
#     """Log accuracy, precision, recall, and F1-Score for all gestures."""
#     for gesture, data in metrics.items():
#         tp, fp, fn = data["tp"], data["fp"], data["fn"]

#         precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#         recall = tp / (tp + fn) if (tp + fn) > 0 else 0
#         f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
#         accuracy = tp / (tp + fp + fn) * 100 if (tp + fp + fn) > 0 else 0

#         logging.info(f"{gesture.upper()} -> Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1_score:.2f}, Accuracy: {accuracy:.2f}%")

# Helper functions for gesture detection
def is_left_click(landmark_list):
    thumb_index_dist = util.get_distance([landmark_list[4], landmark_list[8]])  # Thumb to Index Finger
    return (
        util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
        util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) > 90 and
        thumb_index_dist > 50
    )

def is_right_click(landmark_list):
    thumb_middle_dist = util.get_distance([landmark_list[4], landmark_list[12]])  # Thumb to Middle Finger
    return (
        util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
        util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90 and
        thumb_middle_dist > 50
    )

def is_double_click(landmark_list):
    thumb_index_dist = util.get_distance([landmark_list[4], landmark_list[8]])  # Thumb to Index Finger
    thumb_middle_dist = util.get_distance([landmark_list[4], landmark_list[12]])  # Thumb to Middle Finger
    return (
        util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
        util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
        thumb_index_dist > 50 and
        thumb_middle_dist > 50
    )

def is_screenshot(landmark_list):
    thumb_index_dist = util.get_distance([landmark_list[4], landmark_list[8]])  # Thumb to Index Finger
    return (
        util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
        util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
        thumb_index_dist < 50
    )

def take_screenshot(frame):
    """Take a screenshot and show a notification."""
    screenshot = pyautogui.screenshot()
    label = random.randint(1, 1000)
    screenshot_name = f'my_screenshot_{label}.png'
    screenshot.save(screenshot_name)
    cv2.putText(frame, "Screenshot Taken", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    print(f"Screenshot saved as {screenshot_name}")
    notification.notify(
        title="Screenshot Captured",
        message=f"Saved as {screenshot_name}",
        timeout=5  # Notification disappears after 5 seconds
    )
    print(f"Screenshot saved as {screenshot_name}")

def is_volume_up(landmark_list):
    thumb_ring_dist = util.get_distance([landmark_list[4], landmark_list[16]])  # Thumb to Ring Finger
    return (
        util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 120 and
        util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
        thumb_ring_dist > 50
    )

def is_volume_down(landmark_list):
    thumb_pinky_dist = util.get_distance([landmark_list[4], landmark_list[20]])  # Thumb to Pinky Finger
    return (
        util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
        util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) > 120 and
        thumb_pinky_dist > 50
    )


def is_brightness_up(landmark_list):
    """
    Detects brightness up gesture: Pinky finger raised.
    """
    pinky_tip = landmark_list[20]
    pinky_mcp = landmark_list[17]  # Pinky MCP joint
    return pinky_tip[1] < pinky_mcp[1]  # Pinky tip above MCP joint

def is_brightness_down(landmark_list):
    """
    Detects brightness down gesture: Thumb finger raised.
    """
    thumb_tip = landmark_list[4]
    thumb_mcp = landmark_list[1]  # Thumb MCP joint
    return thumb_tip[0] > thumb_mcp[0]  # Thumb tip to the right of MCP joint (horizontal axis)

    
    # Add conditions for better differentiation
    return (
        70 < util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 120 and
        palm_open_angle > 130 and
        thumb_ring_dist > 50 and
        index_ring_dist < 40
    )


def move_cursor(landmark_list):
    """Move the cursor based on the index finger tip position."""
    # Index finger tip coordinates
    index_finger_tip = landmark_list[8]
    x, y = index_finger_tip[0] * screen_width, index_finger_tip[1] * screen_height

    # Smooth movement
    pyautogui.moveTo(x, y)



def display_metrics_on_frame(frame):
    """Display accuracy for each gesture on the frame."""
    y_offset = 30 
    for gesture, data in metrics.items():
        tp, fp, fn = data["tp"], data["fp"], data["fn"]
        accuracy = tp / (tp + fp + fn) * 100 if (tp + fp + fn) > 0 else 0
        cv2.putText(frame, f"{gesture.capitalize()} Accuracy: {accuracy:.2f}%", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
# gesture_buffer = {"brightness_up": 0, "brightness_down": 0}
# GESTURE_HOLD_THRESHOLD = 5  # Number of frames a gesture must be detected consecutively
def detect_gesture(frame, landmark_list, processed):
    # """Detect gestures and update metrics."""
    # global gesture_buffer
    if len(landmark_list) >= 21:
        if is_left_click(landmark_list):
            update_metrics("left_click", True)
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif is_right_click(landmark_list):
            update_metrics("right_click", True)
            mouse.press(Button.right)
            mouse.release(Button.right)
            cv2.putText(frame, "Right Click", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif is_double_click(landmark_list):
            update_metrics("double_click", True)
            pyautogui.doubleClick()
            cv2.putText(frame, "Double Click", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        elif is_screenshot(landmark_list):
            update_metrics("screenshot", True)
            take_screenshot(frame)
        elif is_volume_up(landmark_list):
            update_metrics("volume_up", True)
            current_volume = volume.GetMasterVolumeLevelScalar()  # Query current volume
            new_volume = min(current_volume + 0.1, 1.0)
            set_volume(new_volume)
            cv2.putText(frame, f"Volume Up: {new_volume:.2f}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif is_volume_down(landmark_list):
            update_metrics("volume_down", True)
            current_volume = volume.GetMasterVolumeLevelScalar()  # Query current volume
            new_volume = max(current_volume - 0.1, 0.0)
            set_volume(new_volume)
            cv2.putText(frame, f"Volume Down: {new_volume:.2f}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        elif is_brightness_up(landmark_list):
            update_metrics("brightness_up", True)
            try:
                current_brightness = sbc.get_brightness(display=0)[0]
                sbc.set_brightness(min(current_brightness + 10, 100))
                cv2.putText(frame, "Brightness Up", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            except Exception as e:
                print(f"Error updating brightness: {e}")
        elif is_brightness_down(landmark_list):
            update_metrics("brightness_down", True)
            try:
                current_brightness = sbc.get_brightness(display=0)[0]
                sbc.set_brightness(max(current_brightness - 10, 0))
                cv2.putText(frame, "Brightness Down", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            except Exception as e:
                print(f"Error updating brightness: {e}")
        else:
            move_cursor(landmark_list)
            cv2.putText(frame, "Moving Cursor", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

def log_metrics():
    """Log accuracy, precision, recall, and F1-Score for all gestures."""
    with open('accuracy_log.txt', 'a') as log_file:
        for gesture, data in metrics.items():
            tp, fp, fn = data["tp"], data["fp"], data["fn"]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = tp / (tp + fp + fn) * 100 if (tp + fp + fn) > 0 else 0

            #logging.info(f"{gesture.upper()} -> Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1_score:.2f}, Accuracy: {accuracy:.2f}%")
            logging.info(f"{gesture.upper()} -> Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1_score:.2f}, Accuracy: {accuracy:.2f}%")
            log_file.write(f"{gesture.upper()} -> Accuracy: {accuracy:.2f}%\n")
def main():
    """Main function to run the application."""
    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)

            landmark_list = []
            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]
                draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    landmark_list.append((lm.x, lm.y))

            detect_gesture(frame, landmark_list, processed)
            display_metrics_on_frame(frame)

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        log_metrics()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()