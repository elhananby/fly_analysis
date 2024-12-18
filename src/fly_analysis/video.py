import cv2
import numpy as np


def threshold(frame: np.ndarray, threshold: int = 127):
    """
    Threshold a frame.

    Parameters:
        frame (np.ndarray): Input frame.
        threshold (int): Threshold value. Default is 127.

    Returns:
        np.ndarray: Thresholded frame.
    """
    return cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY_INV)[1]


def gray(frame: np.ndarray):
    """
    Convert a frame to grayscale.

    Parameters:
        frame (np.ndarray): Input frame.

    Returns:
        np.ndarray: Grayscale frame.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def find_contours(frame: np.ndarray):
    """
    Find contours in a frame.

    Parameters:
        frame (np.ndarray): Input frame.

    Returns:
        np.ndarray: Contours.
    """
    return cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]


def get_largest_contour(contours: np.ndarray):
    """
    Get the largest contour in a list of contours.

    Parameters:
        contours (np.ndarray): Contours.

    Returns:
        np.ndarray: Largest contour.
    """
    return max(contours, key=cv2.contourArea)


def get_contour_parameters(contour: np.ndarray):
    """
    Get parameters of a contour.

    Parameters:
        contour (np.ndarray): Contour.

    Returns:
        centroid: Centroid of the contour.
        area: Area of the contour.
        perimeter: Perimeter of the contour.
        ellipse: Ellipse of the contour.
    """
    M = cv2.moments(contour)

    # calculate x,y coordinate of center
    if M["m00"] == 0:
        return (0, 0), 0, 0, (0, 0, 0)

    centroid = (
        int(M["m10"] / M["m00"]),
        int(M["m01"] / M["m00"]),
    )
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    ellipse = cv2.fitEllipse(contour)
    return centroid, area, perimeter, ellipse


def read_video(filename: str):
    """
    Read a video.

    Parameters:
        filename (str): Path to the video file.

    Returns:
        Iterator[np.ndarray]: Iterator of frames.
    """
    video_reader = cv2.VideoCapture(filename)

    while video_reader.isOpened():
        ret, frame = video_reader.read()
        if ret:
            yield frame
        else:
            break