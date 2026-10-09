import requests
import os

# ESP32_IP = "192.168.0.250"
ESP32_IP = "10.96.42.250"
ESP32_BASE_URL = f"http://{ESP32_IP}"

def set_horizontal_angle(angle: int) -> str:
    """Rotates the ESP32 camera horizontally."""
    try:
        response = requests.post(f"{ESP32_BASE_URL}/horizontal", data=str(angle), timeout=2.0)
        return f"Horizontal servo successfully set to {angle} degrees."
    except Exception as e:
        return f"Failed to set horizontal servo: {e}"

def set_vertical_angle(angle: int) -> str:
    """Rotates the ESP32 camera vertically."""
    try:
        response = requests.post(f"{ESP32_BASE_URL}/vertical", data=str(angle), timeout=2.0)
        return f"Vertical servo successfully set to {angle} degrees."
    except Exception as e:
        return f"Failed to set vertical servo: {e}"
