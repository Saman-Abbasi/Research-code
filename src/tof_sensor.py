# src/tof_sensor.py
# Replaces MiDaS depth estimation with 3x VL53L0X ToF sensors (left/center/right).
# Returns risk values 0.0 (clear) to 1.0 (imminent) per zone — same scale the
# rest of main.py already expects, so SafetyPolicyEngine needs no changes.

import platform
import time

MAX_RANGE_MM = 1200   # beyond this, treat as fully clear (VL53L0X reliable ceiling)
MIN_RANGE_MM = 30      # below this, treat as max risk (sensor's physical floor)

# Pi GPIO pin numbers used to control each sensor's XSHUT pin during address
# assignment. These are PLACEHOLDERS — set the real pin numbers once wired.
XSHUT_PINS = {"left": 17, "center": 27, "right": 22}
NEW_ADDRESSES = {"left": 0x30, "center": 0x31, "right": 0x32}


class ToFArray:
    def __init__(self):
        self.is_real = (platform.system() == "Linux")
        if self.is_real:
            self._init_real_sensors()
        else:
            print("[ToFArray] Non-Linux system detected — using simulated ToF readings.")

    def _init_real_sensors(self):
        import board
        import busio
        import digitalio
        import adafruit_vl53l0x

        i2c = busio.I2C(board.SCL, board.SDA)

        self.xshut = {
            zone: digitalio.DigitalInOut(getattr(board, f"D{pin}"))
            for zone, pin in XSHUT_PINS.items()
        }
        for pin in self.xshut.values():
            pin.switch_to_output(value=False)  # hold all 3 sensors off initially

        self.sensors = {}
        for zone, pin in self.xshut.items():
            pin.value = True       # power up only this one sensor
            time.sleep(0.01)       # let it boot before talking to it over I2C
            sensor = adafruit_vl53l0x.VL53L0X(i2c)
            sensor.set_address(NEW_ADDRESSES[zone])
            self.sensors[zone] = sensor

    def read(self):
        """Returns {'left': risk, 'center': risk, 'right': risk}, each 0.0-1.0."""
        if not self.is_real:
            return {"left": 0.0, "center": 0.0, "right": 0.0}

        readings = {}
        for zone, sensor in self.sensors.items():
            distance_mm = sensor.range
            readings[zone] = self._distance_to_risk(distance_mm)
        return readings

    @staticmethod
    def _distance_to_risk(distance_mm):
        if distance_mm <= MIN_RANGE_MM:
            return 1.0
        if distance_mm >= MAX_RANGE_MM:
            return 0.0
        return 1.0 - (distance_mm - MIN_RANGE_MM) / (MAX_RANGE_MM - MIN_RANGE_MM)