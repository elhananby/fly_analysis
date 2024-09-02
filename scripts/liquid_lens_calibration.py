import numpy as np
from scipy.optimize import curve_fit
import time
import json
import requests
from typing import List, Tuple
from ximea import xiapi
import serial

from lens import Lens

# Constants
DATA_PREFIX = "data: "
FLYDRA2_URL = "http://10.40.80.6:8937"
LENS_DEVICE = "/dev/optotune_ld"
ARDUINO_DEVICE = "/dev/ttyACM0"
ARDUINO_BAUD_RATE = 9600
CAMERA_EXPOSURE = 10000  # in microseconds
STEPS_PER_MM = 200  # Adjust this based on your stepper motor and lead screw

def parse_chunk(chunk: str) -> dict:
    """Parse a chunk of data from the Braidz stream."""
    lines = chunk.strip().split("\n")
    if len(lines) != 2 or lines[0] != "event: braid" or not lines[1].startswith(DATA_PREFIX):
        raise ValueError("Invalid chunk format")
    return json.loads(lines[1][len(DATA_PREFIX):])

def lorentzian(x: np.ndarray, a: float, x0: float, gam: float) -> np.ndarray:
    """Lorentzian distribution function."""
    return a * gam**2 / (gam**2 + (x - x0)**2)

class DeviceManager:
    def __init__(self):
        self.lens = None
        self.arduino = None
        self.braidz = None
        self.camera = None

    def connect_to_lens(self):
        self.lens = Lens(LENS_DEVICE)
        return self.lens.to_focal_power_mode()

    def connect_to_arduino(self):
        self.arduino = serial.Serial(ARDUINO_DEVICE, ARDUINO_BAUD_RATE, timeout=1)
        time.sleep(2)  # Allow time for Arduino to reset
        return self.arduino

    def connect_to_braidz(self):
        session = requests.session()
        if session.get(FLYDRA2_URL).status_code != requests.codes.ok:
            raise ConnectionError("Failed to connect to Flydra2")
        events_url = f"{FLYDRA2_URL}/events"
        self.braidz = session.get(events_url, stream=True, headers={"Accept": "text/event-stream"})
        return self.braidz

    def connect_to_camera(self):
        self.camera = xiapi.Camera()
        print('Opening first camera...')
        self.camera.open_device()
        self.camera.set_exposure(CAMERA_EXPOSURE)
        print(f'Exposure was set to {self.camera.get_exposure()} us')
        return self.camera

class StepperController:
    def __init__(self, arduino: serial.Serial):
        self.arduino = arduino
        self.current_position = 0

    def move_to(self, position_mm: float):
        steps = int((position_mm - self.current_position) * STEPS_PER_MM)
        self.arduino.write(f"MOVE {steps}\n".encode())
        self.wait_for_movement_complete()
        self.current_position = position_mm

    def reset_to_origin(self):
        self.move_to(0)

    def wait_for_movement_complete(self):
        while True:
            if self.arduino.in_waiting:
                response = self.arduino.readline().decode().strip()
                if response == "DONE":
                    break
            time.sleep(0.1)


class CameraController:
    def __init__(self, camera: xiapi.Camera):
        self.camera = camera
        self.img = xiapi.Image()

    def start_acquisition(self):
        print('Starting data acquisition...')
        self.camera.start_acquisition()

    def stop_acquisition(self):
        print('Stopping acquisition...')
        self.camera.stop_acquisition()

    def close_camera(self):
        self.camera.close_device()
        print('Camera closed.')

    def capture_image(self) -> np.ndarray:
        self.camera.get_image(self.img)
        data_numpy = self.img.get_image_data_numpy()
        return data_numpy

class LensController:
    def __init__(self, lens: Lens, camera_controller: CameraController):
        self.lens = lens
        self.camera_controller = camera_controller

    def set_diopter(self, value: float):
        """Set the diopter value of the lens."""
        self.lens.set_diopter(value)

    def perform_sweep(self, start_diopter: float, end_diopter: float, step_size: float) -> Tuple[List[float], List[float]]:
        """Perform a sweep of diopter values and return diopter and contrast values."""
        diopter_values = []
        contrast_values = []
        
        current_diopter = start_diopter
        while current_diopter <= end_diopter:
            self.set_diopter(current_diopter)
            image = self.camera_controller.capture_image()
            contrast = self.calculate_contrast(image)
            
            diopter_values.append(current_diopter)
            contrast_values.append(contrast)
            
            current_diopter += step_size
        
        return diopter_values, contrast_values

    def find_focus(self, min_diopter: float, max_diopter: float) -> float:
        first_sweep_step = 0.4
        second_sweep_step = 0.02
        
        # First sweep
        diopter_values, contrast_values = self.perform_sweep(min_diopter, max_diopter, first_sweep_step)

        # Find diopter with highest contrast from first sweep
        max_contrast_index = np.argmax(contrast_values)
        max_contrast_diopter = diopter_values[max_contrast_index]

        # Define small interval around the max contrast diopter
        interval_min = max(min_diopter, max_contrast_diopter - 0.5)
        interval_max = min(max_diopter, max_contrast_diopter + 0.5)

        # Second sweep
        second_diopter_values, second_contrast_values = self.perform_sweep(interval_min, interval_max, second_sweep_step)

        # Combine data from both sweeps
        diopter_values.extend(second_diopter_values)
        contrast_values.extend(second_contrast_values)

        # Fit Lorentzian distribution to the combined data
        popt, _ = curve_fit(lorentzian, np.array(diopter_values), np.array(contrast_values))
        
        # The x0 parameter of the Lorentzian fit corresponds to the focus point
        return popt[1]

    @staticmethod
    def calculate_contrast(image: np.ndarray) -> float:
        """Calculate the contrast value of a small area close to the image center."""
        if len(image.shape) == 3:
            image = np.mean(image, axis=2)
        
        height, width = image.shape
        center_size = min(height, width) // 5
        
        start_y = (height - center_size) // 2
        end_y = start_y + center_size
        start_x = (width - center_size) // 2
        end_x = start_x + center_size
        
        center_area = image[start_y:end_y, start_x:end_x]
        return np.var(center_area)

class CalibrationManager:
    def __init__(self, device_manager: DeviceManager, lens_controller: LensController, stepper_controller: StepperController):
        self.device_manager = device_manager
        self.lens_controller = lens_controller
        self.stepper_controller = stepper_controller

    def get_z_position(self, r: requests.Response) -> float:
        """Get the current z position from the 3D tracking setup."""
        z_poses = []
        counter = 0

        for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
            data = parse_chunk(chunk)
            try:
                z = data["msg"]["Update"]
                z_poses.append(z)
                counter += 1
            except KeyError:
                pass

            if counter > 100:
                break

        return np.nanmean(z_poses)

    def perform_multi_step_calibration(self, num_steps: int, total_distance: float) -> List[Tuple[float, float]]:
        """Perform multiple calibrations at different z positions."""
        calibration_data = []
        min_fp, max_fp = self.device_manager.connect_to_lens()
        step_size = total_distance / (num_steps - 1)

        for i in range(num_steps):
            current_position = i * step_size
            self.stepper_controller.move_to(current_position)
            
            z_position = self.get_z_position(self.device_manager.braidz)
            best_diopter = self.lens_controller.find_focus(min_fp, max_fp)
            calibration_data.append((z_position, best_diopter))
            
            time.sleep(1)  # Wait for system to stabilize
        
        self.stepper_controller.reset_to_origin()
        return calibration_data

def main():
    total_distance = 300  # Total distance to move in mm
    num_calibration_steps = 30  # Number of calibration steps
    
    device_manager = DeviceManager()
    device_manager.connect_to_arduino()
    device_manager.connect_to_braidz()
    device_manager.connect_to_camera()
    
    camera_controller = CameraController(device_manager.camera)
    lens_controller = LensController(device_manager.lens, camera_controller)
    stepper_controller = StepperController(device_manager.arduino)
    
    calibration_manager = CalibrationManager(device_manager, lens_controller, stepper_controller)
    
    camera_controller.start_acquisition()
    
    calibration_results = calibration_manager.perform_multi_step_calibration(num_calibration_steps, total_distance)
    
    camera_controller.stop_acquisition()
    camera_controller.close_camera()
    
    print("Calibration Results:")
    print("Z Position (mm) | Best Diopter")
    print("-" * 30)
    for z, diopter in calibration_results:
        print(f"{z:14.2f} | {diopter:12.3f}")

if __name__ == "__main__":
    main()
