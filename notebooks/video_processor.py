import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import cv2
cv2.startWindowThread()

pattern = r'.*?(\d+)_obj_id_(\d+)_cam_(\d+)_frame_(\d+)\.[^.]+$'

def normalize_angle(angle):
    """Normalize angle to [-π, π]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def interpolate_heading_circular(df, stim_value):
    # Find the rows where stim_value falls between
    lower_row = df[df['stim_direction'] <= stim_value].iloc[-1]
    upper_row = df[df['stim_direction'] >= stim_value].iloc[0]
    
    # Get the angles
    angle1 = lower_row['heading_direction']
    angle2 = upper_row['heading_direction']
    
    # Calculate the direct angular difference
    diff = angle2 - angle1
    
    # If the difference is greater than π, we need to go the other way around
    if abs(diff) > np.pi:
        if diff > 0:
            angle2 -= 2 * np.pi
        else:
            angle2 += 2 * np.pi
    
    # Calculate interpolation factor
    t = (stim_value - lower_row['stim_direction']) / (upper_row['stim_direction'] - lower_row['stim_direction'])
    
    # Interpolate heading
    heading = angle1 + t * (angle2 - angle1)
    
    # Normalize the result back to [-π, π]
    return normalize_angle(heading)

def calculate_velocities(df, dt=0.002):  # dt = 1/500 for 500Hz
    # Calculate velocities using gradient with explicit time step
    df['xvel'] = np.gradient(df['x'], dt)  # pixels per second
    df['yvel'] = np.gradient(df['y'], dt)  # pixels per second
    
    # Calculate heading (velocity direction in degrees)
    df['heading'] = np.degrees(np.arctan2(df['yvel'], df['xvel']))
        
    return df

def process_video(video_path, show_video=False):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize numpy array with NaN values
    results = np.full((total_frames, 4), np.nan)
    # Fill the first column with frame numbers
    results[:, 0] = np.arange(total_frames)
    
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Try to fit an ellipse
            if len(largest_contour) >= 5:  # Need at least 5 points to fit an ellipse
                ellipse = cv2.fitEllipse(largest_contour)
                
                # Extract information
                (x, y), (MA, ma), angle = ellipse
                
                # Update the corresponding row in results
                results[frame_number, 1:] = [x, y, angle]
            
        frame_number += 1
    
    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()
    
    # Convert results to a pandas DataFrame with headers
    df = pd.DataFrame(results, columns=['frame', 'x', 'y', 'orientation'])
    
    # Calculate velocities and heading
    df = calculate_velocities(df)

    return df


def sg_smooth(arr):
    return savgol_filter(arr, 51, 3)

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process video files')
    parser.add_argument('input', type=str, help='Input video file')

    args = parser.parse_args()

    process_video(args.input, show_video=True)