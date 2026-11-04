"""
Pico W firmware used with the online estimation (end-to-end trials): the same as
main_collection.py, but paced at about 2800 Hz (one reading every 357 us, i.e.
2801 Hz), close to the 2800 Hz grid of the online pipeline. The Raspberry Pi still
timestamps each value on reception and interpolates it onto that grid.
Copy to the Pico W as main.py.
"""

import machine
import utime

# Configuration
ADC_PIN = 26
adc = machine.ADC(ADC_PIN)

# LED to indicate running
led = machine.Pin("LED", machine.Pin.OUT)

# Target sample rate of the online pipeline (online/online_rul_piezo.py)
TARGET_SR = 2800
# Calculate the period in microseconds
PERIOD_US = 1000000 // TARGET_SR

def sample_loop():
    """Read the ADC every PERIOD_US microseconds and print each value (0-65535)."""
    led.on()

    # Get the initial time
    next_sample_time = utime.ticks_us()

    try:
        while True:
            # Block until it is exactly time for the next sample
            while utime.ticks_diff(utime.ticks_us(), next_sample_time) < 0:
                pass

            # Advance the target time strictly (prevents drift)
            next_sample_time = utime.ticks_add(next_sample_time, PERIOD_US)

            # Read 16-bit unsigned (0-65535) and print immediately
            val = adc.read_u16()
            print(val)

    except KeyboardInterrupt:
        led.off()

if __name__ == "__main__":
    # Blink 3 times to signal ready
    for _ in range(3):
        led.on()
        utime.sleep(0.1)
        led.off()
        utime.sleep(0.1)

    sample_loop()
