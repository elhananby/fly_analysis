import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import base64
from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
import time

app = Flask(__name__)
socketio = SocketIO(app)


def detect_fly(frame):
    """Detect the fly in the frame using thresholding and contour detection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        ellipse = cv2.fitEllipse(largest_contour)
        return ellipse, largest_contour
    return None, None


def estimate_heading(ellipse, contour):
    """Estimate the heading of the fly based on shape analysis."""
    (x, y), (MA, ma), angle = ellipse

    hull = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull)

    if defects is not None:
        max_defect = max(defects, key=lambda x: x[0][3])
        far_point = tuple(contour[max_defect[0][2]][0])

        dx = far_point[0] - x
        dy = far_point[1] - y
        heading = np.arctan2(dy, dx)
    else:
        heading = np.radians(angle)

    return heading, far_point


def draw_visualization(frame, ellipse, contour, heading, far_point):
    """Draw visualization elements on the frame."""
    # Draw contour
    cv2.drawContours(frame, [contour], 0, (0, 255, 0), 2)

    # Draw ellipse
    cv2.ellipse(frame, ellipse, (255, 0, 0), 2)

    # Draw heading line
    (x, y), _, _ = ellipse
    heading_end = (int(x + 50 * np.cos(heading)), int(y + 50 * np.sin(heading)))
    cv2.line(frame, (int(x), int(y)), heading_end, (0, 0, 255), 2)

    # Draw far point (potential head location)
    cv2.circle(frame, far_point, 3, (255, 255, 0), -1)

    return frame


def process_video(video_path, visualize="none"):
    """Process the video and extract fly trajectory and heading information."""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    results = []
    prev_pos = None
    prev_heading = None

    for frame_num in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        ellipse, contour = detect_fly(frame)
        if ellipse is None:
            continue

        (x, y), (MA, ma), angle = ellipse
        heading, far_point = estimate_heading(ellipse, contour)

        if prev_pos is not None:
            velocity = np.sqrt((x - prev_pos[0]) ** 2 + (y - prev_pos[1]) ** 2) * fps
            if prev_heading is not None:
                angular_velocity = (heading - prev_heading) * fps
            else:
                angular_velocity = 0
        else:
            velocity = 0
            angular_velocity = 0

        results.append(
            {
                "frame": frame_num,
                "x": x,
                "y": y,
                "heading": heading,
                "velocity": velocity,
                "angular_velocity": angular_velocity,
                "major_axis": MA,
                "minor_axis": ma,
            }
        )

        if visualize == "browser":
            vis_frame = draw_visualization(
                frame.copy(), ellipse, contour, heading, far_point
            )
            _, buffer = cv2.imencode(".jpg", vis_frame)
            img_str = base64.b64encode(buffer).decode("utf-8")
            socketio.emit("update_image", {"image": img_str, "frame": frame_num})
            time.sleep(1 / fps)  # Slow down to actual video speed

    cap.release()
    return pd.DataFrame(results)


def detect_behaviors(df):
    """Detect specific behaviors like backward flight or sharp turns."""
    df["backward_flight"] = (df["velocity"] > 0) & (
        np.abs(np.sin(df["heading"] - np.arctan2(df["y"].diff(), df["x"].diff()))) > 0.5
    )

    df["sharp_turn"] = np.abs(df["angular_velocity"]) > np.radians(
        45
    )  # Adjust threshold as needed

    return df


@app.route("/")
def index():
    return render_template("index.html")


def start_server():
    socketio.run(app, host="0.0.0.0", port=5000)


def main(video_path, visualize="none"):
    """Main function to process video and save results."""
    if visualize == "browser":
        server_thread = threading.Thread(target=start_server)
        server_thread.start()
        print(
            "Server started. Please open a web browser and navigate to http://localhost:5000"
        )
        time.sleep(2)  # Give the server a moment to start

    df = process_video(video_path, visualize)
    df = detect_behaviors(df)

    output_path = Path(video_path).with_suffix(".csv")
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze fly behavior in video.")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument(
        "-v",
        "--visualize",
        choices=["none", "browser"],
        default="none",
        help="Visualization method",
    )
    args = parser.parse_args()

    main(args.video_path, args.visualize)
