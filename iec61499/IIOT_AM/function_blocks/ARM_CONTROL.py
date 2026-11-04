from adafruit_servokit import ServoKit
import xml.etree.ElementTree as ETree
import sys
import os
import time
import threading

# --- GLOBAL LOCK & CONTROLLER ---
# We use a global variable so we don't re-initialize the I2C bus (which causes crashes)
if not hasattr(sys, "dinasore_pca9685_kit"):
    sys.dinasore_pca9685_kit = None
    sys.dinasore_pca9685_lock = threading.Lock()

def read_configuration(config_name):
    moves = dict()
    sequence_list = []
    
    # Robust path finding
    base_path = os.path.dirname(sys.path[0])
    config_path = os.path.join(base_path, 'resources', 'function_blocks', 'arm_config', config_name + '.xml')
    
    # Fallback if running outside standard structure
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'arm_config', config_name + '.xml')

    if os.path.exists(config_path):
        try:
            tree = ETree.parse(config_path)
            root = tree.getroot()
            for info in root:
                if info.tag == 'Position':
                    # Parse position data
                    moves[info.attrib['Name']] = [float(info.attrib['Base']),
                                                  float(info.attrib['Left']),
                                                  float(info.attrib['Right']),
                                                  float(info.attrib['Gripper'])]

                elif info.tag == 'Sequences':
                    for sequence in info:
                        if sequence.tag == 'Sequence':
                            sequence_list.append([sequence.attrib['position_name'], float(sequence.attrib['time'])])
        except Exception as e:
            print(f"[ERROR] XML Parse Error: {e}")
            
    return moves, sequence_list

class ARM_CONTROL:
    def __init__(self):
        self.moves = None
        self.sequence_list = None
        
        # Initialize Hardware ONCE using the Global Singleton pattern
        with sys.dinasore_pca9685_lock:
            if sys.dinasore_pca9685_kit is None:
                try:
                    print("[HARDWARE] Initializing PCA9685 I2C Board (0x40)...")
                    # Channels=16 is standard for the Hat
                    sys.dinasore_pca9685_kit = ServoKit(channels=16)
                except ValueError:
                    print("[FATAL] I2C Device 0x40 not found! Check connection/power.")
                except OSError:
                    print("[FATAL] I2C Error. Run 'sudo modprobe i2c_bcm2835' to enable driver.")

    def relax_servos(self):
        """Turns off the PWM signal so motors stop fighting and cool down."""
        if sys.dinasore_pca9685_kit:
            try:
                # Setting pulse width to 0 often relaxes digital servos
                # Alternatively, you can just stop sending the signal
                for i in range(16):
                    try:
                        sys.dinasore_pca9685_kit.servo[i].angle = None 
                    except:
                        pass
            except Exception as e:
                print(f"[WARN] Could not relax servos: {e}")

    def set_pos(self, pos, sleep_time):
        if sys.dinasore_pca9685_kit is None:
            print("[ERROR] Cannot move: Hardware not initialized.")
            return

        base, left, right, claw = pos
        kit = sys.dinasore_pca9685_kit
        
        try:
            # Map your specific servo calibration here
            # BASE (Channel 0)
            kit.servo[0].set_pulse_width_range(500, 2500)
            kit.servo[0].actuation_range = 180
            kit.servo[0].angle = base
            
            # LEFT (Channel 1)
            kit.servo[1].set_pulse_width_range(1150, 2400)
            kit.servo[1].actuation_range = 120
            kit.servo[1].angle = 120 - left  # inverted mapping of the left servo
            
            # RIGHT (Channel 2)
            kit.servo[2].set_pulse_width_range(600, 1600)
            kit.servo[2].actuation_range = 120
            kit.servo[2].angle = right
            
            # CLAW (Channel 3)
            kit.servo[3].set_pulse_width_range(500, 2000)
            kit.servo[3].actuation_range = 130
            kit.servo[3].angle = claw
            
            time.sleep(sleep_time)
            self.relax_servos()
            
        except Exception as e:
            print(f"[ERROR] Servo I2C Write Failed: {e}")

    def schedule(self, event_name, event_value, file_name):
        
        if event_name == 'INIT':
            self.moves, self.sequence_list = read_configuration(file_name)
            
            # Initial Move
            if self.sequence_list:
                position_name, seq_time = self.sequence_list[0]
                print(f"Moving to position {position_name} - {seq_time}s")
                self.set_pos(self.moves[position_name], seq_time)
            
            return [event_value, None]

        elif event_name == 'RUN':
            if sys.dinasore_pca9685_kit is None:
                print("[ERROR] Hardware missing. INIT failed.")
                return [None, None]

            if self.sequence_list:
                for sequence in self.sequence_list:
                    position_name, seq_time = sequence
                    print(f"Moving to position {position_name} - {seq_time}s")
                    self.set_pos(self.moves[position_name], seq_time)
            
            self.relax_servos()

            return [None, event_value]