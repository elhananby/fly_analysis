import cv2
import numpy as np
import pandas as pd
import argparse
import os
import glob
import re

def draw_arrow(frame, start_point, end_point, color):
    if not (np.isnan(start_point[0]) or np.isnan(start_point[1]) or 
            np.isnan(end_point[0]) or np.isnan(end_point[1])):
        cv2.arrowedLine(frame, 
                        (int(start_point[0]), int(start_point[1])), 
                        (int(end_point[0]), int(end_point[1])), 
                        color, 2, tipLength=0.2)
    return frame

def process_video(video_path, csv_path, output_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Define color for the arrow
    arrow_color = (0, 255, 0)  # Green

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise ValueError(f"Error creating output video file: {output_path}")
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get keypoints for the current frame
        frame_data = df[df['frame_idx'] == frame_count]
        
        if not frame_data.empty:
            abdomen_point = (frame_data['abdomen.x'].values[0], frame_data['abdomen.y'].values[0])
            head_point = (frame_data['head.x'].values[0], frame_data['head.y'].values[0])
            
            # Draw arrow from abdomen to head
            frame = draw_arrow(frame, abdomen_point, head_point, arrow_color)
        
        # Write the frame
        out.write(frame)
        
        frame_count += 1
        
        # Display progress
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames")
    
    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def get_matching_files(video_path, csv_path):
    if os.path.isfile(video_path) and os.path.isfile(csv_path):
        return [(video_path, csv_path)]
    
    video_files = glob.glob(os.path.join(video_path, "*.mp4"))
    csv_files = glob.glob(os.path.join(csv_path, "*.csv"))
    
    matched_files = []
    for video_file in video_files:
        video_id = re.search(r'obj_id_(\d+)_frame_(\d+)', os.path.basename(video_file))
        if video_id:
            obj_id, frame = video_id.groups()
            matching_csv = next((csv for csv in csv_files if f'obj_id_{obj_id}_frame_{frame}' in csv), None)
            if matching_csv:
                matched_files.append((video_file, matching_csv))
    
    return matched_files

def main():
    parser = argparse.ArgumentParser(description="Generate videos with fly orientation arrows.")
    parser.add_argument("video_input", help="Path to the input video file or folder")
    parser.add_argument("csv_input", help="Path to the CSV file or folder containing keypoint data")
    parser.add_argument("--output", help="Path for the output video file or folder (optional)")
    
    args = parser.parse_args()
    
    matched_files = get_matching_files(args.video_input, args.csv_input)
    
    if not matched_files:
        print("No matching video and CSV files found.")
        return
    
    for video_file, csv_file in matched_files:
        if args.output:
            if os.path.isdir(args.output):
                output_file = os.path.join(args.output, os.path.basename(video_file).replace('.mp4', '_tracked.mp4'))
            else:
                output_file = args.output
        else:
            output_file = video_file.replace('.mp4', '_tracked.mp4')
        
        try:
            print(f"Processing {video_file} with {csv_file}")
            process_video(video_file, csv_file, output_file)
            print(f"Completed processing {video_file}")
        except Exception as e:
            print(f"An error occurred while processing {video_file}: {str(e)}")
    
    print("All video processing complete!")

if __name__ == "__main__":
    main()