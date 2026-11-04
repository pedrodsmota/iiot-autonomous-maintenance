import serial
import time
import csv
import sys
import signal
import threading
import queue
import os

# Configuration
SERIAL_PORT = '/dev/ttyACM0'  # Adjust if it appears as ACM1
BAUD_RATE = 115200          # USB CDC ignores baud rate usually, but good practice
DURATION = 3600               # Seconds to record (0 = unlimited, use Ctrl+C to stop)
OUTPUT_FILE = 'piezo_data.csv'

# Buffer configuration
BUFFER_SIZE = 4000  # Samples per buffer (~1 second at typical rates)
NUM_BUFFERS = 3     # Triple buffering

# Global state
stop_requested = False
disk_error = False
total_samples_written = 0
writer_queue = queue.Queue(maxsize=NUM_BUFFERS)

def signal_handler(sig, frame):
    global stop_requested
    print("\nStop requested. Finishing up...")
    stop_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def writer_thread_func(output_file):
    """Background thread that writes buffers to disk."""
    global disk_error, total_samples_written
    
    try:
        # Try to set CPU affinity to core 1 (Linux only)
        os.sched_setaffinity(0, {1})
    except (AttributeError, OSError):
        pass  # Not available on all systems
    
    try:
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time_Sec', 'Vibration_Val'])
            
            while True:
                try:
                    # Wait for a buffer with timeout
                    buffer = writer_queue.get(timeout=0.5)
                    
                    if buffer is None:  # Shutdown signal
                        break
                    
                    # Write buffer to file
                    try:
                        writer.writerows(buffer)
                        f.flush()
                        total_samples_written += len(buffer)
                        print(f"  [Writer] Wrote {len(buffer)} samples (total: {total_samples_written})")
                    except (IOError, OSError) as e:
                        print(f"\n*** DISK WRITE ERROR: {e} ***")
                        print("*** Stopping to preserve data integrity ***")
                        disk_error = True
                        break
                        
                except queue.Empty:
                    if stop_requested:
                        # Drain remaining buffers
                        while not writer_queue.empty():
                            try:
                                buffer = writer_queue.get_nowait()
                                if buffer is not None:
                                    writer.writerows(buffer)
                                    f.flush()
                                    total_samples_written += len(buffer)
                            except queue.Empty:
                                break
                        break
                        
    except (IOError, OSError) as e:
        print(f"\n*** DISK ERROR opening file: {e} ***")
        disk_error = True


def capture_data():
    global stop_requested, disk_error
    
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Connected to {SERIAL_PORT}")
    except serial.SerialException as e:
        print(f"Error connecting to Pico: {e}")
        print("Check if the Pico is connected and script is running.")
        return

    # Flush any stale data
    ser.reset_input_buffer()

    # Try to set CPU affinity to core 0 for main thread (Linux only)
    try:
        os.sched_setaffinity(0, {0})
    except (AttributeError, OSError):
        pass

    # Start writer thread
    writer = threading.Thread(target=writer_thread_func, args=(OUTPUT_FILE,), daemon=True)
    writer.start()

    duration_str = f"{DURATION} seconds" if DURATION > 0 else "indefinitely (Ctrl+C to stop)"
    print(f"Recording vibration data for {duration_str}...")
    print(f"Using triple-buffered writing (buffer size: {BUFFER_SIZE} samples)")
    
    # Current buffer being filled
    current_buffer = []
    total_samples_collected = 0
    buffers_sent = 0
    
    start_time = time.time()
    last_print = start_time
    
    try:
        while not stop_requested and not disk_error:
            # Check duration limit
            elapsed = time.time() - start_time
            if DURATION > 0 and elapsed >= DURATION:
                break
                
            if ser.in_waiting:
                # Read line, strip whitespace
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                
                if line.isdigit():
                    val = int(line)
                    now = time.time()
                    current_buffer.append((now - start_time, val))
                    total_samples_collected += 1
                    
                    # Buffer full? Send to writer thread
                    if len(current_buffer) >= BUFFER_SIZE:
                        try:
                            writer_queue.put(current_buffer, timeout=1.0)
                            buffers_sent += 1
                            print(f"Buffer {buffers_sent} filled ({total_samples_collected} samples collected)")
                            current_buffer = []
                        except queue.Full:
                            print("\nWARNING: Writer thread too slow! Dropping buffer.")
                            current_buffer = []
                
            # Print status every second
            if time.time() - last_print > 1.0:
                print(f"Collected {total_samples_collected} samples...", end='\r')
                last_print = time.time()
                
    except KeyboardInterrupt:
        stop_requested = True
    finally:
        ser.close()

    end_time = time.time()
    actual_duration = end_time - start_time
    
    # Send remaining samples
    if current_buffer and not disk_error:
        try:
            writer_queue.put(current_buffer, timeout=2.0)
        except queue.Full:
            print("WARNING: Could not save final buffer!")
    
    # Signal writer thread to stop
    try:
        writer_queue.put(None, timeout=1.0)
    except queue.Full:
        pass
    
    # Wait for writer to finish
    print("\nWaiting for writer thread to finish...")
    writer.join(timeout=5.0)
    
    if total_samples_collected == 0:
        print("\nNo data collected.")
        return

    fs = total_samples_collected / actual_duration
    print(f"\n=== Summary ===")
    print(f"Total Time: {actual_duration:.2f}s")
    print(f"Total Samples Collected: {total_samples_collected}")
    print(f"Total Samples Written: {total_samples_written}")
    print(f"Effective Sample Rate: {fs:.2f} Hz")
    print(f"Buffers Sent: {buffers_sent}")
    
    if disk_error:
        print("\n*** WARNING: Disk error occurred! Some data may be lost. ***")
    else:
        print(f"\nData saved to {OUTPUT_FILE}")
    
    print("Done.")

if __name__ == '__main__':
    capture_data()
