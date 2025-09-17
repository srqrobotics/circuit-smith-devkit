import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Dimensions dictionary (in mm)
# ----------------------------
sensor_dimensions = {
    'BMP280 Module': {'length': 20, 'width': 18, 'height': 3},
    'A3144 Hall Effect Sensor Module': {'length': 30, 'width': 20, 'height': 5},
    'CO2 Sensor Module (MH-Z19)': {'length': 60, 'width': 40, 'height': 30},
    'UV Sensor Module (ML8511)': {'length': 25, 'width': 20, 'height': 5},
    'buzzer': {'length': 30, 'width': 30, 'height': 10},
    'Rtc Module': {'length': 30, 'width': 15, 'height': 7},
    'RCWL-0516 Microwave Motion Sensor': {'length': 46, 'width': 36, 'height': 8},
    'SX1278 LORA Module': {'length': 49, 'width': 25, 'height': 13},
    'Flame Sensor Module': {'length': 35, 'width': 30, 'height': 10},
    'Capacitive Touch Sensor Module (TTP223)': {'length': 26, 'width': 18, 'height': 5},
    'PIR Motion Sensor': {'length': 75, 'width': 55, 'height': 45},
    'MPU 6050': {'length': 20, 'width': 15, 'height': 3},
    'KY-008 Laser Diode Module': {'length': 25, 'width': 20, 'height': 10},
    'Stepper Motor Driver Module A4988': {'length': 40, 'width': 25, 'height': 10},
    'OLED Display': {'length': 50, 'width': 40, 'height': 3},
    'Sound Sensor Module (KY-038)': {'length': 35, 'width': 30, 'height': 10},
    'SD card Module': {'length': 25, 'width': 20, 'height': 7},
    'ov7670-camera-module': {'length': 30, 'width': 30, 'height': 5},
    'Keypad Module': {'length': 50, 'width': 50, 'height': 7},
    'L298 Motor Driver': {'length': 50, 'width': 30, 'height': 10},
    'DS18B20 Waterproof Temperature Sensor': {'length': 60, 'width': 10, 'height': 10},
    'Gas': {'length': 45, 'width': 30, 'height': 10},
    'BME280 Temperature, Humidity, and Pressure Sensor': {'length': 20, 'width': 18, 'height': 3},
    'AS608-Optical-Fingerprint-Sensor': {'length': 60, 'width': 40, 'height': 18},
    'I2C LCD': {'length': 80, 'width': 40, 'height': 15},
    'MPU9250 9-DOF IMU Module': {'length': 30, 'width': 25, 'height': 5},
    'DHT22': {'length': 50, 'width': 30, 'height': 10},
    'servo-SG90': {'length': 22, 'width': 11.5, 'height': 31},
    'Gesture Sensor Module (APDS-9960)': {'length': 23, 'width': 18, 'height': 5},
    'MAX30100 Sensor': {'length': 30, 'width': 30, 'height': 7},
    'MAX7219 Dot Matrix Display Module': {'length': 70, 'width': 70, 'height': 10},
    'HMC5883L': {'length': 22, 'width': 18, 'height': 5},
    'RF Receiver Module (433 MHz)': {'length': 35, 'width': 20, 'height': 10},
    'HX711 Load Cell Amp': {'length': 44, 'width': 22, 'height': 8},
    'NEO 6M GPS Module': {'length': 40, 'width': 30, 'height': 12},
    'NRF24L01+ Wireless Transceiver Module': {'length': 50, 'width': 25, 'height': 6},
    'Soil Moisture Sensor Module': {'length': 50, 'width': 20, 'height': 10},
    'KY-040 Rotary Encoder Module': {'length': 25, 'width': 25, 'height': 10},
    'MAX6675 Thermocouple Temperature Sensor Module': {'length': 40, 'width': 30, 'height': 10},
    'Relay': {'length': 30, 'width': 30, 'height': 20},
    'RF Transmitter Module (433 MHz)': {'length': 35, 'width': 20, 'height': 10},
    'LDR Light Sensor Module': {'length': 35, 'width': 25, 'height': 5},
    'RFID RC522': {'length': 40, 'width': 30, 'height': 10},
    'TCS3200 Color Detection Sensor Module': {'length': 40, 'width': 30, 'height': 10},
    'HC-05 BT Module': {'length': 32, 'width': 18, 'height': 7},
    'ADXL345 Accelerometer Module': {'length': 30, 'width': 25, 'height': 5},
    'IR Receiver': {'length': 25, 'width': 20, 'height': 10},
    'ultrasonic-SR04': {'length': 45, 'width': 30, 'height': 15},
    'INA219 Current Sensor Module': {'length': 35, 'width': 20, 'height': 8},
    'DFPlayer Mini MP3 Player Module': {'length': 35, 'width': 25, 'height': 7}
}

# ----------------------------
# Configuration
# ----------------------------
PX_PER_MM = 10  # You can change this for higher/lower resolution

# ----------------------------
# Helpers
# ----------------------------
def show_image_with_grid(png_path, grid_divisions):
    image = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not load image: {png_path}")

    height, width = image.shape[:2]
    grid_size_x = width // grid_divisions
    grid_size_y = height // grid_divisions

    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_xticks(np.arange(0, width, grid_size_x))
    ax.set_yticks(np.arange(0, height, grid_size_y))
    ax.grid(color='r', linestyle='-', linewidth=0.5)
    plt.title(os.path.basename(png_path))
    plt.show()

def resize_image_and_adjust_coordinates(png_path, json_path, scale_ratio, output_png_path, output_json_path):
    image = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not load image: {png_path}")

    new_size = (
        int(image.shape[1] * scale_ratio),
        int(image.shape[0] * scale_ratio)
    )

    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(output_png_path, resized_image)

    with open(json_path, 'r') as f:
        data = json.load(f)

    if "digital-pins" in data and "reloc" in data["digital-pins"]:
        for pin in data["digital-pins"]["reloc"]:
            pin["points"][0] = int(pin["points"][0] * scale_ratio)
            pin["points"][1] = int(pin["points"][1] * scale_ratio)

    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"✔ Saved: {output_png_path}, {output_json_path}")

def get_devices_from_folder(folder_path):
    files = os.listdir(folder_path)
    json_files = {f[:-5] for f in files if f.endswith('.json')}
    png_files = {f[:-4] for f in files if f.endswith('.png')}
    return sorted(json_files.intersection(png_files))

def process_device(device, folder_path, sensor_dimensions, px_per_mm=PX_PER_MM):
    image_path = os.path.join(folder_path, f"{device}.png")
    json_path = os.path.join(folder_path, f"{device}.json")

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"⚠ Skipping {device} (image not found)")
        return

    if device not in sensor_dimensions:
        print(f"⚠ Skipping {device} (no dimension data)")
        return

    image_height, image_width = image.shape[:2]
    real_dims = sensor_dimensions[device]

    # Decide which is the longer side
    real_mm = max(real_dims['length'], real_dims['width'])
    image_px = max(image_width, image_height)

    scale_ratio = (real_mm * px_per_mm) / image_px

    print(f"➡ {device}: Scaling by {scale_ratio:.2f}x to match {real_mm} mm width")

    # Optional debug
    show_image_with_grid(image_path, 10)

    output_folder = "dev-boards"
    os.makedirs(output_folder, exist_ok=True)

    resize_image_and_adjust_coordinates(
        image_path,
        json_path,
        scale_ratio,
        os.path.join(output_folder, f"{device}.png"),
        os.path.join(output_folder, f"{device}.json")
    )

# ----------------------------
# Main
# ----------------------------
if __name__ == '__main__':
    input_folder = "ref"
    devices = get_devices_from_folder(input_folder)

    print(f"\n🧩 Found {len(devices)} devices to process...\n")

    for device in devices:
        process_device(device, input_folder, sensor_dimensions)
