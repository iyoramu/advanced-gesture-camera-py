#!/usr/bin/env python3
"""
Professional Gesture-Controlled Camera System
- Uses OpenCV and MediaPipe for hand tracking
- Implements state machine for camera control
- Features gesture-based photo capture
- Includes camera focus control
- Demonstrates clean architecture suitable for top tech companies
"""

import cv2
import numpy as np
import time
import os
from datetime import datetime
import mediapipe as mp
from typing import Tuple, List, Optional, Dict

class GestureControlledCamera:
    def __init__(self, 
                 output_dir: str = "captures",
                 camera_index: int = 0,
                 focus_delay: float = 1.5,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize gesture-controlled camera system.
        
        Args:
            output_dir: Directory to save captured images
            camera_index: Index of camera device
            focus_delay: Delay before capture after focus gesture (seconds)
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
        """
        self.output_dir = output_dir
        self.camera_index = camera_index
        self.focus_delay = focus_delay
        self.focus_start_time = 0
        self.capture_count = 0
        self.camera_state = "IDLE"  # States: IDLE, FOCUSING, READY_TO_CAPTURE
        
        # MediaPipe hands configuration
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Camera setup
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {self.camera_index}")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        
    def __del__(self):
        """Release resources when object is destroyed."""
        self.cap.release()
        cv2.destroyAllWindows()
    
    def _get_timestamp(self) -> str:
        """Generate formatted timestamp for filenames."""
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    def _save_image(self, frame: np.ndarray) -> str:
        """Save image to output directory with timestamp."""
        filename = os.path.join(self.output_dir, f"capture_{self._get_timestamp()}.jpg")
        cv2.imwrite(filename, frame)
        return filename
    
    def _calculate_fps(self) -> float:
        """Calculate and display frames per second."""
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed
        return fps
    
    def _count_fingers(self, landmarks: List, handedness: str) -> int:
        """
        Count extended fingers based on hand landmarks.
        
        Args:
            landmarks: MediaPipe hand landmarks
            handedness: "Left" or "Right" hand
            
        Returns:
            Number of extended fingers (0-5)
        """
        finger_tips = [self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
                      self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                      self.mp_hands.HandLandmark.RING_FINGER_TIP,
                      self.mp_hands.HandLandmark.PINKY_TIP]
        
        finger_mcp = [self.mp_hands.HandLandmark.INDEX_FINGER_MCP,
                     self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
                     self.mp_hands.HandLandmark.RING_FINGER_MCP,
                     self.mp_hands.HandLandmark.PINKY_MCP]
        
        thumb_tip = self.mp_hands.HandLandmark.THUMB_TIP
        thumb_ip = self.mp_hands.HandLandmark.THUMB_IP
        
        count = 0
        
        # Check thumb (different logic for left/right hand)
        if handedness == "Right":
            if landmarks[thumb_tip].x > landmarks[thumb_ip].x:
                count += 1
        else:
            if landmarks[thumb_tip].x < landmarks[thumb_ip].x:
                count += 1
                
        # Check other fingers
        for tip, mcp in zip(finger_tips, finger_mcp):
            if landmarks[tip].y < landmarks[mcp].y:
                count += 1
                
        return count
    
    def _process_gestures(self, frame: np.ndarray, results) -> Dict[str, int]:
        """
        Process hand gestures and return finger counts.
        
        Args:
            frame: Input camera frame
            results: MediaPipe hand detection results
            
        Returns:
            Dictionary with finger counts for each hand
        """
        gesture_info = {"Left": 0, "Right": 0}
        
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, 
                                                results.multi_handedness):
                # Get hand side (left/right)
                hand_label = handedness.classification[0].label
                
                # Count fingers
                finger_count = self._count_fingers(hand_landmarks.landmark, hand_label)
                gesture_info[hand_label] = finger_count
                
                # Draw hand landmarks (for visualization)
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Display finger count
                cv2.putText(frame, f"{hand_label}: {finger_count}", 
                          (10, 30 if hand_label == "Left" else 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return gesture_info
    
    def _update_camera_state(self, gesture_info: Dict[str, int]) -> None:
        """
        Update camera state machine based on gestures.
        
        State transitions:
        - IDLE -> FOCUSING: When two fingers are shown (either hand)
        - FOCUSING -> READY_TO_CAPTURE: After focus delay
        - READY_TO_CAPTURE -> CAPTURING: When one finger is shown
        - Any state -> IDLE: When no hands are detected
        """
        total_fingers = gesture_info["Left"] + gesture_info["Right"]
        
        if total_fingers == 0:
            self.camera_state = "IDLE"
        elif self.camera_state == "IDLE" and total_fingers >= 2:
            self.camera_state = "FOCUSING"
            self.focus_start_time = time.time()
        elif self.camera_state == "FOCUSING" and (time.time() - self.focus_start_time) > self.focus_delay:
            self.camera_state = "READY_TO_CAPTURE"
        elif self.camera_state == "READY_TO_CAPTURE" and total_fingers == 1:
            self.camera_state = "CAPTURING"
            self.capture_count = max(gesture_info["Left"], gesture_info["Right"])
    
    def _display_status(self, frame: np.ndarray) -> None:
        """Display current system status on the frame."""
        fps = self._calculate_fps()
        
        # Display state and FPS
        cv2.putText(frame, f"State: {self.camera_state}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Visual feedback for states
        if self.camera_state == "FOCUSING":
            progress = min(1.0, (time.time() - self.focus_start_time) / self.focus_delay)
            cv2.rectangle(frame, (0, 0), 
                         (int(frame.shape[1] * progress), 20),
                         (0, 255, 0), -1)
        elif self.camera_state == "READY_TO_CAPTURE":
            cv2.putText(frame, "READY - Show 1 finger to capture", 
                      (frame.shape[1]//4, frame.shape[0] - 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def run(self):
        """Main application loop."""
        print("Gesture Controlled Camera System - Press 'q' to quit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error reading frame from camera")
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hand detection
            results = self.hands.process(rgb_frame)
            
            # Analyze gestures
            gesture_info = self._process_gestures(frame, results)
            
            # Update camera state machine
            self._update_camera_state(gesture_info)
            
            # Handle capture action
            if self.camera_state == "CAPTURING":
                for i in range(self.capture_count):
                    filename = self._save_image(frame)
                    print(f"Captured image: {filename}")
                    time.sleep(0.3)  # Small delay between multiple captures
                self.camera_state = "IDLE"
            
            # Display system status
            self._display_status(frame)
            
            # Show the frame
            cv2.imshow('Gesture Controlled Camera', frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        print("System shutdown")

def main():
    """Entry point for the application."""
    try:
        camera = GestureControlledCamera(
            output_dir="gesture_captures",
            camera_index=0,
            focus_delay=1.0
        )
        camera.run()
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0

if __name__ == "__main__":
    main()
