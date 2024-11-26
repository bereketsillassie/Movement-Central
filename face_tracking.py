# face_tracking.py

import numpy as np
import time
import math

# PID Controller Parameters
w, h = 640, 480  # Frame dimensions for visualization (can be adjusted)
center = w // 2
# Updated PID coefficients: [Kp, Ki, Kd]
pid_yaw = [0.5, 0.0001, 0.25]  # Adjusted PID coefficients for yaw
pid_fb = [0.6, 0.0001, 0.1]    # Adjusted PID coefficients for forward/backward
pError_yaw = 0
iError_yaw = 0  # Integral error for yaw
dError_yaw = 0  # Derivative error for yaw

pError_fb = 0
iError_fb = 0  # Integral error for forward/backward
dError_fb = 0  # Derivative error for forward/backward

desired_face_area = 5000  # Desired area of the face in pixels
center_offset = 0  # Offset for the center of the frame

# Configuration Flags
INVERT_YAW_CONTROL = False  # Set to True if robot moves opposite to desired direction

def track_face(info, pid_yaw, pid_fb, pError_yaw, iError_yaw, dError_yaw,
               pError_fb, iError_fb, dError_fb, desired_face_area, center_offset):
    """
    PID control for face tracking.
    Returns the command string and updated PID errors.
    """
    # Unpack info
    face_center = info[0]
    area = info[1]
    x, y = face_center[0], face_center[1]
    center_x = (w // 2) + center_offset

    # Yaw PID
    error_yaw = x - center_x
    iError_yaw += error_yaw
    # Limit integral error to prevent windup
    max_iError_yaw = 10000  # Adjust as necessary
    iError_yaw = max(-max_iError_yaw, min(max_iError_yaw, iError_yaw))
    dError_yaw = error_yaw - pError_yaw
    speed_yaw = (pid_yaw[0] * error_yaw) + (pid_yaw[1] * iError_yaw) + (pid_yaw[2] * dError_yaw)
    max_speed_yaw = 30.0
    speed_yaw = np.clip(speed_yaw, -max_speed_yaw, max_speed_yaw)  # Limit steering angle

    # Invert speed_yaw if necessary
    if INVERT_YAW_CONTROL:
        speed_yaw = -speed_yaw

    # Forward/Backward PID
    error_fb = desired_face_area - area
    iError_fb += error_fb
    # Limit integral error to prevent windup
    max_iError_fb = 10000  # Adjust as necessary
    iError_fb = max(-max_iError_fb, min(max_iError_fb, iError_fb))
    dError_fb = error_fb - pError_fb
    if area > 0:
        speed_fb = (pid_fb[0] * error_fb) + (pid_fb[1] * iError_fb) + (pid_fb[2] * dError_fb)
    else:
        speed_fb = 0
    max_speed_fb = 1.0  # Adjust as appropriate
    speed_fb = np.clip(speed_fb, -max_speed_fb, max_speed_fb)  # Limit speed

    print(f"[DEBUG] Face Center X: {x}, Frame Center X: {center_x}")
    print(f"[DEBUG] Error Yaw: {error_yaw}, IError Yaw: {iError_yaw}, DError Yaw: {dError_yaw}")
    print(f"[DEBUG] Speed Yaw: {speed_yaw}, Error FB: {error_fb}, IError FB: {iError_fb}, DError FB: {dError_fb}")
    print(f"[DEBUG] Speed FB: {speed_fb}")

    pError_yaw = error_yaw
    pError_fb = error_fb

    # Map steering angle to side-side command (0-126, with 64 as center)
    side_side_command = int(64 + (speed_yaw / max_speed_yaw) * 62)
    side_side_command = max(0, min(126, side_side_command))

    # Map desired speed to front-back command (0-126, with 64 as stop)
    front_back_command = int(64 + (speed_fb / max_speed_fb) * 62)
    front_back_command = max(0, min(126, front_back_command))

    command_string = f"{front_back_command} {side_side_command}"
    print(f"[DEBUG] Side Side Command: {side_side_command}, Front Back Command: {front_back_command}")
    print(f"[DEBUG] Command sent: {command_string}")

    return command_string, pError_yaw, iError_yaw, dError_yaw, pError_fb, iError_fb, dError_fb

def face_tracking_process(command_queue, client):
    global desired_face_area, center_offset
    global pid_yaw, pid_fb, pError_yaw, iError_yaw, dError_yaw
    global pError_fb, iError_fb, dError_fb

    while True:
        if not command_queue.empty():
            command, data = command_queue.get()
            if command == 'face_tracking':
                detection = data
                if 'detections' in detection and len(detection['detections']) > 0:
                    # Use the first detection for control
                    detection = detection['detections'][0]
                    info = [detection['center'], detection['area']]

                    # Use PID controller to get command
                    command_string, pError_yaw, iError_yaw, dError_yaw, \
                        pError_fb, iError_fb, dError_fb = track_face(
                            info, pid_yaw, pid_fb, pError_yaw, iError_yaw, dError_yaw,
                            pError_fb, iError_fb, dError_fb, desired_face_area, center_offset
                        )

                    # Publish the command
                    client.publish("robot/control", command_string)
                else:
                    # No detection received; stop the robot
                    front_back_command = 64  # Stop
                    side_side_command = 64   # Neutral steering
                    command_string = f"{front_back_command} {side_side_command}"
                    print("Stopping (no detection received)")
                    client.publish("robot/control", command_string)
            elif command == 'increase_face_area':
                desired_face_area += 100
                desired_face_area = int(np.clip(desired_face_area, 3000, 9000))
                print(f"Desired face area increased to {desired_face_area}")
            elif command == 'decrease_face_area':
                desired_face_area -= 100
                desired_face_area = int(np.clip(desired_face_area, 3000, 9000))
                print(f"Desired face area decreased to {desired_face_area}")
            elif command == 'move_center_left':
                center_offset -= 10
                center_offset = int(np.clip(center_offset, -320, 320))
                print(f"Center offset moved left to {center_offset}")
            elif command == 'move_center_right':
                center_offset += 10
                center_offset = int(np.clip(center_offset, -320, 320))
                print(f"Center offset moved right to {center_offset}")
            elif command == 'update_pid':
                kp, ki, kd = data
                pid_yaw = [kp, ki, kd]
                pid_fb = [kp, ki, kd]
                print(f"Updated PID parameters: Kp={kp}, Ki={ki}, Kd={kd}")
        else:
            time.sleep(0.05)
