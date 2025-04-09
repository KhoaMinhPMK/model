import numpy as np
import time

class EmergencySignalDetector:
    def __init__(self):
        self.emergency_pose_start = None
        self.emergency_pose_threshold = 2  # Thời gian mặc định cho tư thế khẩn cấp
        self.above_head_threshold = 4  # Thời gian dài hơn cho tay giơ cao quá đầu
        self.emergency_active = False
        self.arm_angle_threshold = 100  # Maximum angle for emergency pose
        self.head_offset = 50  # Pixels above head to consider "raised"
        self.head_side_threshold = 30  # Pixels buffer for head level
        self.shoulder_threshold = 20  # Pixels buffer for shoulder level
        self.eye_threshold = 30  # Pixels buffer for eye level
        self.wrist_threshold = 15  # Vùng đệm cho vị trí cổ tay
        self.current_threshold = None

    def calculate_arm_angle(self, shoulder, elbow, wrist):
        """Calculate angle at elbow between upper and lower arm"""
        v1 = np.array([shoulder[0] - elbow[0], shoulder[1] - elbow[1]])
        v2 = np.array([wrist[0] - elbow[0], wrist[1] - elbow[1]])
        
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        
        angle = np.arccos(np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0))
        return np.degrees(angle)

    def check_arm_position(self, wrist, nose, shoulder, angle):
        """Check wrist position and return appropriate signal type"""
        head_top = nose[1] - self.head_offset
        eye_level = nose[1]
        shoulder_level = shoulder[1]
        wrist_y = wrist[1]
        
        if wrist_y <= head_top:
            return "above_head"
        elif wrist_y <= eye_level:
            return "above_head"
        elif wrist_y <= shoulder_level:
            return "above_head"
        
        if angle < self.arm_angle_threshold:
            return "bent_raised"
        
        return None

    def check_emergency_pose(self, keypoints):
        """Check if person is holding emergency pose"""
        if len(keypoints) == 0:
            return False
            
        nose = keypoints[0]
        left_shoulder = keypoints[5]
        left_elbow = keypoints[7]
        left_wrist = keypoints[9]
        right_shoulder = keypoints[6]
        right_elbow = keypoints[8]
        right_wrist = keypoints[10]
        
        key_points = [nose, left_shoulder, left_elbow, left_wrist,
                     right_shoulder, right_elbow, right_wrist]
        if any(kp[2] < 0.5 for kp in key_points):
            return False

        # Check left arm
        left_angle = self.calculate_arm_angle(left_shoulder, left_elbow, left_wrist)
        left_position = self.check_arm_position(left_wrist, nose, left_shoulder, left_angle)
        
        # Check right arm
        right_angle = self.calculate_arm_angle(right_shoulder, right_elbow, right_wrist)
        right_position = self.check_arm_position(right_wrist, nose, right_shoulder, right_angle)
        
        if left_position == "above_head" or right_position == "above_head":
            self.current_threshold = self.above_head_threshold
            return True
        elif left_position == "bent_raised" or right_position == "bent_raised":
            self.current_threshold = self.emergency_pose_threshold
            return True
            
        return False

    def update_emergency_status(self, keypoints, current_state):
        """Update and return emergency status"""
        if current_state not in ["stand", "sleep"]:
            if self.check_emergency_pose(keypoints):
                if self.emergency_pose_start is None:
                    self.emergency_pose_start = time.time()
                    self.current_threshold = self.emergency_pose_threshold
                elif time.time() - self.emergency_pose_start >= self.current_threshold:
                    self.emergency_active = True
                    return True
            else:
                self.emergency_pose_start = None
                self.emergency_active = False
        
        return False
