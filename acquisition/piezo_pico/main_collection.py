"""
Pico W firmware used to record the dataset (sessions 1-22): reads the ADC as fast
as possible and prints each value on its own line over USB serial. The rate is set
by the serial link and the host, about 2760-2900 Hz depending on the session; the Raspberry Pi
timestamps each value on reception (capture_serial.py).
Copy to the Pico W as main.py.
"""

import machine
import utime

# Configuration
ADC_PIN = 26
adc = machine.ADC(ADC_PIN)

# LED to indicate running
led = machine.Pin("LED", machine.Pin.OUT)

def sample_loop():
    """Read the ADC in a tight loop and print each value (0-65535) on its own line."""
    led.on()
    try:
        while True:
            print(adc.read_u16())   # no pacing: the rate is limited by the USB serial link
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
