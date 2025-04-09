from collections import deque
import numpy as np
import time
# Keep but comment out
# from emergency_signal_detector import EmergencySignalDetector

class StatusAnalyzer:
    def __init__(self, memory_size=5, fall_threshold=3):
        self.memory_size = memory_size
        self.fall_threshold = fall_threshold
        self.fall_memory = deque(maxlen=memory_size)
        self.motion_memory = deque(maxlen=memory_size)  # Add motion memory
        self.prev_box = None
        self.no_motion_start_time = None
        self.no_motion_threshold = 10  # seconds
        self.preparation_start_time = None
        self.preparation_threshold = 5  # seconds for preparation phase
        self.alert_ready = False
        self.motion_threshold = 10  # Define motion threshold
        self.last_motion_time = time.time()
        # Keep but comment out
        # self.emergency_detector = EmergencySignalDetector()
        self.emergency_active = False  # Add this line to fix the error

    def check_overlap(self, person_box, pillow_box):
        """Check if person and pillow boxes overlap"""
        p_x1, p_y1, p_x2, p_y2 = person_box
        pl_x1, pl_y1, pl_x2, pl_y2 = pillow_box
        
        # Calculate overlap area
        x_left = max(p_x1, pl_x1)
        y_top = max(p_y1, pl_y1)
        x_right = min(p_x2, pl_x2)
        y_bottom = min(p_y2, pl_y2)
        
        if x_right < x_left or y_bottom < y_top:
            return False
            
        # Calculate overlap ratio
        overlap_area = (x_right - x_left) * (y_bottom - y_top)
        person_area = (p_x2 - p_x1) * (p_y2 - p_y1)
        overlap_ratio = overlap_area / person_area
        
        # Return True if overlap ratio is significant (e.g., > 20%)
        return overlap_ratio > 0.2
    
    def check_parallel_overlap(self, person_box, bed_box):
        """Check if person is parallel to and overlapping with bed"""
        p_x1, p_y1, p_x2, p_y2 = person_box
        b_x1, b_y1, b_x2, b_y2 = bed_box
        
        # Calculate person and bed dimensions
        person_width = p_x2 - p_x1
        person_height = p_y2 - p_y1
        bed_width = b_x2 - b_x1
        
        # Check if person is lying (width > height)
        is_lying = person_width > person_height
        
        # Calculate overlap as before
        x_left = max(p_x1, b_x1)
        y_top = max(p_y1, b_y1)
        x_right = min(p_x2, b_x2)
        y_bottom = min(p_y2, b_y2)
        
        if x_right < x_left or y_bottom < y_top:
            return False
            
        overlap_area = (x_right - x_left) * (y_bottom - y_top)
        person_area = person_width * person_height
        overlap_ratio = overlap_area / person_area
        
        # Return True if person is lying and has significant overlap (>30%)
        return is_lying and overlap_ratio > 0.3
    
    def calculate_motion(self, current_box):
        """Calculate motion between current and previous box"""
        if self.prev_box is None:
            self.prev_box = current_box
            return 0
        
        # Calculate center points
        curr_center = [(current_box[0] + current_box[2])/2, 
                      (current_box[1] + current_box[3])/2]
        prev_center = [(self.prev_box[0] + self.prev_box[2])/2, 
                      (self.prev_box[1] + self.prev_box[3])/2]
        
        # Calculate motion as Euclidean distance
        motion = np.sqrt((curr_center[0] - prev_center[0])**2 + 
                        (curr_center[1] - prev_center[1])**2)
        
        # Update motion timing
        if motion > self.motion_threshold:  # Use class motion threshold
            self.last_motion_time = time.time()
            self.no_motion_start_time = None
            self.preparation_start_time = None
            self.alert_ready = False
        elif self.no_motion_start_time is None:
            self.no_motion_start_time = time.time()
            
        self.prev_box = current_box
        return motion

    def analyze_status(self, person_boxes, pillow_boxes, bed_boxes, current_state, keypoints=None):
        """Analyze if person is sleeping, falling, or showing emergency signal"""
        # Keep emergency signal check but commented out
        # if keypoints is not None:
        #     if self.emergency_detector.update_emergency_status(keypoints, current_state):
        #         self.emergency_active = self.emergency_detector.emergency_active
        #         return "fall_alert"

        # Normal status analysis
        if "lie" in current_state and len(person_boxes) > 0:
            for person_box in person_boxes:
                # Check bed overlap first
                for bed_box in bed_boxes:
                    if self.check_parallel_overlap(person_box, bed_box):
                        self.fall_memory.clear()
                        return "sleep"
                
                # If no bed overlap, check pillow overlap
                for pillow_box in pillow_boxes:
                    if self.check_overlap(person_box, pillow_box):
                        self.fall_memory.clear()
                        return "sleep"
        
        # Only track actual falls, not like_fall states
        if not self.emergency_active:
            if "like_fall" in current_state:
                self.fall_memory.append("normal")  # Don't count like_fall as fall
            elif current_state == "lie":
                self.fall_memory.append("fall")
            else:
                self.fall_memory.append("normal")
                
            fall_count = sum(1 for x in self.fall_memory if x == "fall")
            
            # Calculate average motion if we have motion memory
            if len(person_boxes) > 0:
                current_motion = self.calculate_motion(person_boxes[0])
                self.motion_memory.append(current_motion)
                avg_motion = sum(self.motion_memory) / len(self.motion_memory) if self.motion_memory else 0

                if len(self.fall_memory) >= 3 and fall_count >= self.fall_threshold:
                    if avg_motion < self.motion_threshold:  # Use class motion threshold
                        current_time = time.time()
                        
                        # Check initial no motion duration (10s)
                        if (self.no_motion_start_time and 
                            current_time - self.no_motion_start_time >= self.no_motion_threshold):
                            
                            # Start preparation phase if not started
                            if not self.preparation_start_time:
                                self.preparation_start_time = current_time
                                self.alert_ready = False
                                return "fall_prepare"  # New intermediate state
                            
                            # Check if preparation time has elapsed (5s)
                            if current_time - self.preparation_start_time >= self.preparation_threshold:
                                self.alert_ready = True
                                return "fall_alert"
                            
                            return "fall_prepare"
                            
                        return "fall"  # Still in initial no-motion period
                    
                    self.no_motion_start_time = None
                    self.preparation_start_time = None
                    self.alert_ready = False
                    return "fall"
            
        return "normal"

    def get_preparation_time(self):
        """Get elapsed time in preparation phase"""
        if self.preparation_start_time:
            return time.time() - self.preparation_start_time
        return 0

    def get_memory_status(self):
        """Get current status of fall memory"""
        return list(self.fall_memory)
