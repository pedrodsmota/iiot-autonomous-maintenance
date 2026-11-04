#define _GNU_SOURCE
#define _DEFAULT_SOURCE
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <sched.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/types.h>
#include <linux/spi/spidev.h>
#include <time.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <getopt.h>

/*
 * MPU9250 SPI Driver for Raspberry Pi
 * Target Sample Rate: 4 kHz using FIFO (Polling mode, no Interrupt pin)
 * Uses double-buffering and a writer thread to save data without blocking sampling.
 */

#define SPI_DEVICE "/dev/spidev0.0"
#define SPI_SPEED 4000000

// MPU9250 Registers
#define REG_PWR_MGMT_1    0x6B
#define REG_PWR_MGMT_2    0x6C
#define REG_USER_CTRL     0x6A
#define REG_CONFIG        0x1A
#define REG_ACCEL_CONFIG  0x1C
#define REG_ACCEL_CONFIG2 0x1D
#define REG_FIFO_EN       0x23
#define REG_FIFO_COUNTH   0x72
#define REG_FIFO_COUNTL   0x73
#define REG_FIFO_R_W      0x74
#define REG_WHO_AM_I      0x75

// Bits
#define BIT_H_RESET       0x80
#define BIT_I2C_IF_DIS    0x10
#define BIT_FIFO_EN       0x40
#define BIT_FIFO_RST      0x04
#define BIT_ACCEL_FIFO_EN 0x08

// SPI Command Flag
#define READ_FLAG         0x80

// Buffer configuration
// Each buffer holds ~1 second of data at 4kHz = 4000 samples
// 4000 samples * 6 bytes = 24KB per buffer (very manageable)
#define BUFFER_SAMPLES 4000

typedef struct {
    int16_t x;
    int16_t y;
    int16_t z;
} AccelRaw;

typedef struct {
    AccelRaw samples[BUFFER_SAMPLES];
    int count;
    int buffer_id;  // Sequential ID for ordering
} SampleBuffer;

// Double buffer system
#define NUM_BUFFERS 3  // Triple buffering for extra safety
SampleBuffer buffers[NUM_BUFFERS];
atomic_int buffer_write_idx = 0;      // Index of buffer being written to disk
atomic_int buffer_fill_idx = 0;       // Index of buffer being filled
atomic_int buffers_ready[NUM_BUFFERS] = {0, 0, 0};  // Flag: buffer ready to write

// Writer thread control
atomic_int writer_running = 1;
atomic_int total_buffers_written = 0;
pthread_t writer_thread;
pthread_mutex_t writer_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t writer_cond = PTHREAD_COND_INITIALIZER;

// Statistics
atomic_int total_samples_collected = 0;
atomic_int fifo_overflows = 0;
atomic_int disk_write_error = 0;  // Flag for disk full/write errors

// Global state
int spi_fd = -1;
FILE *csv_file = NULL;
volatile sig_atomic_t stop_requested = 0;

void signal_handler(int sig) {
    (void)sig;
    stop_requested = 1;
    fprintf(stderr, "\nStop requested. Finishing up...\n");
}

void exit_error(const char *msg) {
    perror(msg);
    if (spi_fd >= 0) close(spi_fd);
    if (csv_file) fclose(csv_file);
    exit(1);
}

void mpu_write_reg(uint8_t reg, uint8_t value) {
    uint8_t tx[] = {reg, value};
    uint8_t rx[2] = {0, 0};
    
    struct spi_ioc_transfer tr = {
        .tx_buf = (unsigned long)tx,
        .rx_buf = (unsigned long)rx,
        .len = 2,
        .speed_hz = SPI_SPEED,
        .bits_per_word = 8,
    };

    if (ioctl(spi_fd, SPI_IOC_MESSAGE(1), &tr) < 0)
        exit_error("SPI Write failed");
}

uint8_t mpu_read_reg(uint8_t reg) {
    uint8_t tx[] = {reg | READ_FLAG, 0x00};
    uint8_t rx[2] = {0, 0};
    
    struct spi_ioc_transfer tr = {
        .tx_buf = (unsigned long)tx,
        .rx_buf = (unsigned long)rx,
        .len = 2,
        .speed_hz = SPI_SPEED,
        .bits_per_word = 8,
    };

    if (ioctl(spi_fd, SPI_IOC_MESSAGE(1), &tr) < 0)
        exit_error("SPI Read failed");
    
    return rx[1];
}

uint16_t mpu_read_u16(uint8_t reg) {
    uint8_t tx[] = {reg | READ_FLAG, 0x00, 0x00};
    uint8_t rx[3] = {0, 0, 0};
    
    struct spi_ioc_transfer tr = {
        .tx_buf = (unsigned long)tx,
        .rx_buf = (unsigned long)rx,
        .len = 3,
        .speed_hz = SPI_SPEED,
        .bits_per_word = 8,
    };

    if (ioctl(spi_fd, SPI_IOC_MESSAGE(1), &tr) < 0)
        exit_error("SPI Read u16 failed");
    
    return (rx[1] << 8) | rx[2];
}

void mpu_read_burst(uint8_t reg, uint8_t *dest, int len) {
    uint8_t *tx_buf = calloc(len + 1, sizeof(uint8_t));
    uint8_t *rx_buf = calloc(len + 1, sizeof(uint8_t));
    
    if (!tx_buf || !rx_buf) exit_error("Memory allocation failed");
    
    tx_buf[0] = reg | READ_FLAG;
    
    struct spi_ioc_transfer tr = {
        .tx_buf = (unsigned long)tx_buf,
        .rx_buf = (unsigned long)rx_buf,
        .len = len + 1,
        .speed_hz = SPI_SPEED * 4,
        .bits_per_word = 8,
    };

    if (ioctl(spi_fd, SPI_IOC_MESSAGE(1), &tr) < 0)
        exit_error("SPI Burst Read failed");
    
    memcpy(dest, &rx_buf[1], len);
    
    free(tx_buf);
    free(rx_buf);
}

// Writer thread function - runs on separate core
void *writer_thread_func(void *arg) {
    (void)arg;
    
    // Try to pin to CPU 1 (sampling will be on CPU 0)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(1, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    
    while (writer_running || total_buffers_written < atomic_load(&buffer_fill_idx)) {
        int idx = atomic_load(&buffer_write_idx) % NUM_BUFFERS;
        
        // Wait for buffer to be ready
        pthread_mutex_lock(&writer_mutex);
        while (!atomic_load(&buffers_ready[idx]) && writer_running) {
            // Wait with timeout to check writer_running periodically
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_nsec += 100000000; // 100ms timeout
            if (ts.tv_nsec >= 1000000000) {
                ts.tv_sec++;
                ts.tv_nsec -= 1000000000;
            }
            pthread_cond_timedwait(&writer_cond, &writer_mutex, &ts);
        }
        pthread_mutex_unlock(&writer_mutex);
        
        if (!atomic_load(&buffers_ready[idx])) {
            if (!writer_running) break;
            continue;
        }
        
        // Write buffer to file
        SampleBuffer *buf = &buffers[idx];
        int write_error = 0;
        for (int i = 0; i < buf->count && !write_error; i++) {
            float x_g = buf->samples[i].x / 2048.0f;
            float y_g = buf->samples[i].y / 2048.0f;
            float z_g = buf->samples[i].z / 2048.0f;
            if (fprintf(csv_file, "%d,%.4f,%.4f,%.4f\n", 
                    buf->buffer_id * BUFFER_SAMPLES + i + 1, x_g, y_g, z_g) < 0) {
                write_error = 1;
            }
        }
        
        if (fflush(csv_file) != 0 || write_error) {
            fprintf(stderr, "\n*** DISK WRITE ERROR (disk full?) ***\n");
            fprintf(stderr, "*** Stopping to preserve data integrity ***\n");
            atomic_store(&disk_write_error, 1);
            stop_requested = 1;  // Signal main loop to stop
            break;
        }
        
        // Mark buffer as available
        atomic_store(&buffers_ready[idx], 0);
        atomic_fetch_add(&buffer_write_idx, 1);
        total_buffers_written++;
        
        fprintf(stderr, "  [Writer] Buffer %d written (%d samples)\n", buf->buffer_id, buf->count);
    }
    
    return NULL;
}

int main(int argc, char *argv[]) {
    int target_duration = 0;
    const char *output_file = NULL;
    int opt;
    
    // Parse arguments
    while ((opt = getopt(argc, argv, "s:t:f:")) != -1) {
        switch (opt) {
            case 's':
                output_file = optarg;
                break;
            case 't':
                target_duration = atoi(optarg);
                break;
            case 'f':
                // Frequency is currently fixed in this driver, ignoring the flag
                fprintf(stderr, "Note: -f flag is ignored for mpu9250, frequency is fixed ~4kHz.\n");
                break;
            default:
                fprintf(stderr, "Usage: %s [-s output_file] [-t duration] [-f frequency_ignored]\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }
    
    if (target_duration > 0) {
        fprintf(stderr, "Will record for %d seconds\n", target_duration);
    } else {
        fprintf(stderr, "Recording indefinitely. Press Ctrl+C to stop.\n");
    }
    
    // Setup signal handler for graceful shutdown
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Try to pin main thread to CPU 0
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    
    // 1. Open SPI Device
    spi_fd = open(SPI_DEVICE, O_RDWR);
    if (spi_fd < 0) exit_error("Failed to open SPI device");

    uint8_t mode = SPI_MODE_3;
    if (ioctl(spi_fd, SPI_IOC_WR_MODE, &mode) < 0) exit_error("Failed to set SPI mode");
    
    uint8_t bits = 8;
    if (ioctl(spi_fd, SPI_IOC_WR_BITS_PER_WORD, &bits) < 0) exit_error("Failed to set bits per word");
    
    uint32_t speed = SPI_SPEED;
    if (ioctl(spi_fd, SPI_IOC_WR_MAX_SPEED_HZ, &speed) < 0) exit_error("Failed to set speed");

    // 2. Init MPU9250
    fprintf(stderr, "Initializing MPU9250...\n");
    
    mpu_write_reg(REG_PWR_MGMT_1, BIT_H_RESET);
    usleep(100000);
    
    uint8_t who = mpu_read_reg(REG_WHO_AM_I);
    fprintf(stderr, "WhoAmI: 0x%02X (Expected: 0x71)\n", who);
    if (who != 0x71) {
        fprintf(stderr, "WARNING: Unexpected WhoAmI value. Check wiring.\n");
    }

    mpu_write_reg(REG_USER_CTRL, BIT_I2C_IF_DIS);
    usleep(1000);

    mpu_write_reg(REG_PWR_MGMT_1, 0x01);
    usleep(1000);

    mpu_write_reg(REG_ACCEL_CONFIG2, 0x08);
    mpu_write_reg(REG_ACCEL_CONFIG, 0x18);

    mpu_write_reg(REG_USER_CTRL, BIT_I2C_IF_DIS);
    mpu_write_reg(REG_FIFO_EN, BIT_ACCEL_FIFO_EN);
    
    fprintf(stderr, "MPU9250 Configured.\n");
    
    // Initialize buffers
    for (int i = 0; i < NUM_BUFFERS; i++) {
        buffers[i].count = 0;
        buffers[i].buffer_id = -1;
    }
    
    // Open CSV file
    if (output_file == NULL || strcmp(output_file, "-") == 0 || strcmp(output_file, "/dev/stdout") == 0) {
        csv_file = stdout;
    } else {
        csv_file = fopen(output_file, "w");
        if (!csv_file) exit_error("Failed to open CSV file");
    }
    fprintf(csv_file, "Sample,Accel_X,Accel_Y,Accel_Z\n");
    
    // Start writer thread
    if (pthread_create(&writer_thread, NULL, writer_thread_func, NULL) != 0) {
        exit_error("Failed to create writer thread");
    }
    
    fprintf(stderr, "Starting capture with triple-buffering...\n");
    fprintf(stderr, "Buffer size: %d samples (~1 second each)\n", BUFFER_SAMPLES);

    // Reset FIFO
    mpu_write_reg(REG_USER_CTRL, BIT_I2C_IF_DIS | BIT_FIFO_RST);
    usleep(100);
    mpu_write_reg(REG_USER_CTRL, BIT_I2C_IF_DIS | BIT_FIFO_EN);

    struct timespec start_time, current_time;
    clock_gettime(CLOCK_REALTIME, &start_time);
    fprintf(stderr, "Start Time: %s", ctime(&start_time.tv_sec));

//------
    struct timespec last_heartbeat_time;
    clock_gettime(CLOCK_REALTIME, &last_heartbeat_time);
    int consecutive_zero_reads = 0;
//------

    // Data Loop
    const int bytes_per_sample = 6;
    uint8_t data_buf[1024];
    
    int skip_sample = 0;
    int current_buffer_idx = 0;
    int buffer_count = 0;
    SampleBuffer *current_buffer = &buffers[0];
    current_buffer->count = 0;
    current_buffer->buffer_id = 0;
    
    while (!stop_requested) {
        // Check duration limit if set
        if (target_duration > 0) {
            clock_gettime(CLOCK_REALTIME, &current_time);
            double elapsed = (current_time.tv_sec - start_time.tv_sec) + 
                           (current_time.tv_nsec - start_time.tv_nsec) / 1e9;
            if (elapsed >= target_duration) {
                fprintf(stderr, "\nTarget duration reached.\n");
                break;
            }
        }

        // --- ADD THIS HEARTBEAT CHECK ---
        clock_gettime(CLOCK_REALTIME, &current_time);
        double hb_elapsed = (current_time.tv_sec - last_heartbeat_time.tv_sec) + 
                            (current_time.tv_nsec - last_heartbeat_time.tv_nsec) / 1e9;
        
        // Run the sanity check every 1.0 seconds
        if (hb_elapsed >= 1.0) {
            uint8_t who = mpu_read_reg(REG_WHO_AM_I);
            uint8_t fifo_status = mpu_read_reg(REG_FIFO_EN);
            
            // If the sensor disappears OR loses its settings, trigger auto-recovery
            if (who != 0x71 || !(fifo_status & BIT_ACCEL_FIFO_EN)) {
                
                if (who != 0x71) {
                    fprintf(stderr, "\n[WARNING] SPI Drop at %.2f sec! (WHO_AM_I: 0x%02X). Waiting for wire to reconnect...\n", hb_elapsed, who);
                } else {
                    fprintf(stderr, "\n[WARNING] Sensor Reset Detected at %.2f sec! Re-initializing...\n", hb_elapsed);
                }
                
                // Wait 50ms to allow the arm to move past the "bad angle" and the wire to reconnect
                usleep(50000); 
                
                // --- AUTO-RECOVERY SEQUENCE ---
                
                // 1. Hardware Reset (brings the chip back to a known state if it's glitching)
                mpu_write_reg(REG_PWR_MGMT_1, BIT_H_RESET);
                usleep(10000); // Wait 10ms for reset
                
                // 2. Wake up & Disable I2C (crucial for SPI stability)
                mpu_write_reg(REG_USER_CTRL, BIT_I2C_IF_DIS);
                mpu_write_reg(REG_PWR_MGMT_1, 0x01); // Auto clock selection
                usleep(1000);
                
                // 3. Reconfigure Accelerometer (your specific 4kHz settings)
                mpu_write_reg(REG_ACCEL_CONFIG2, 0x08);
                mpu_write_reg(REG_ACCEL_CONFIG, 0x18);
                
                // 4. Re-enable FIFO
                mpu_write_reg(REG_USER_CTRL, BIT_I2C_IF_DIS);
                mpu_write_reg(REG_FIFO_EN, BIT_ACCEL_FIFO_EN);
                
                // 5. Reset FIFO to clear any garbage data caused by the static on the broken wire
                mpu_write_reg(REG_USER_CTRL, BIT_I2C_IF_DIS | BIT_FIFO_RST);
                usleep(100);
                mpu_write_reg(REG_USER_CTRL, BIT_I2C_IF_DIS | BIT_FIFO_EN);
                
                // Reset heartbeat timer and error counters
                clock_gettime(CLOCK_REALTIME, &last_heartbeat_time);
                consecutive_zero_reads = 0; 
                
                fprintf(stderr, "[RECOVERY SUCCESSFUL] Resuming data capture...\n");
                continue; // Jump back to the start of the while loop to resume sampling
            }
            
            last_heartbeat_time = current_time; // Reset the timer on a successful check
        }
        // --------------------------------
        
        uint16_t fifo_count = mpu_read_u16(REG_FIFO_COUNTH);
        
        if (fifo_count & 0x8000 || fifo_count >= 512) {
            fprintf(stderr, "\nFIFO Overflow! Resetting. (Lost samples)\n");
            atomic_fetch_add(&fifo_overflows, 1);
            mpu_write_reg(REG_USER_CTRL, BIT_I2C_IF_DIS | BIT_FIFO_RST);
            usleep(100);
            mpu_write_reg(REG_USER_CTRL, BIT_I2C_IF_DIS | BIT_FIFO_EN);
            continue;
        }

        if (fifo_count >= bytes_per_sample) {
            int samples_to_read = fifo_count / bytes_per_sample;
            int bytes_to_read = samples_to_read * bytes_per_sample;
            
            if (bytes_to_read > (int)sizeof(data_buf)) {
                bytes_to_read = (sizeof(data_buf) / bytes_per_sample) * bytes_per_sample;
                samples_to_read = bytes_to_read / bytes_per_sample;
            }

            mpu_read_burst(REG_FIFO_R_W, data_buf, bytes_to_read);
            
            for (int i = 0; i < samples_to_read; i++) {
                // Software Decimation 8kHz -> 4kHz
                if (skip_sample) {
                    skip_sample = 0;
                    continue;
                }
                skip_sample = 1;

                int off = i * 6;
                int idx = current_buffer->count;
                current_buffer->samples[idx].x = (int16_t)((data_buf[off+0] << 8) | data_buf[off+1]);
                current_buffer->samples[idx].y = (int16_t)((data_buf[off+2] << 8) | data_buf[off+3]);
                current_buffer->samples[idx].z = (int16_t)((data_buf[off+4] << 8) | data_buf[off+5]);
                current_buffer->count++;
                atomic_fetch_add(&total_samples_collected, 1);
                
                // Buffer full? Swap to next
                if (current_buffer->count >= BUFFER_SAMPLES) {
                    // Signal writer thread
                    atomic_store(&buffers_ready[current_buffer_idx], 1);
                    pthread_mutex_lock(&writer_mutex);
                    pthread_cond_signal(&writer_cond);
                    pthread_mutex_unlock(&writer_mutex);
                    
                    buffer_count++;
                    fprintf(stderr, "Buffer %d filled (%d total samples so far)\n", 
                           current_buffer->buffer_id, atomic_load(&total_samples_collected));
                    
                    // Move to next buffer
                    current_buffer_idx = (current_buffer_idx + 1) % NUM_BUFFERS;
                    current_buffer = &buffers[current_buffer_idx];
                    
                    // Wait if next buffer is still being written (shouldn't happen with triple buffer)
                    int wait_count = 0;
                    while (atomic_load(&buffers_ready[current_buffer_idx])) {
                        usleep(1000);
                        wait_count++;
                        if (wait_count > 1000) {
                            fprintf(stderr, "ERROR: Writer thread too slow! Buffer overflow.\n");
                            break;
                        }
                    }
                    
                    current_buffer->count = 0;
                    current_buffer->buffer_id = buffer_count;
                    atomic_store(&buffer_fill_idx, buffer_count);
                }
            }
            consecutive_zero_reads = 0; // <-- ADD THIS LINE to reset the counter on a good read
        }

        else {
            consecutive_zero_reads++;
            if (consecutive_zero_reads > 5000) { // Roughly 2.5 seconds of no data
                fprintf(stderr, "\n[CRITICAL ERROR] MPU9250 stopped filling the FIFO (EMI Noise likely).\n");
                stop_requested = 1;
                continue;
            }
        }
        
        usleep(500);  // Slightly faster polling
    }

    // Finalize
    fprintf(stderr, "\nStopping capture...\n");
    
    // Mark last partial buffer as ready if it has data
    if (current_buffer->count > 0) {
        atomic_store(&buffers_ready[current_buffer_idx], 1);
        atomic_store(&buffer_fill_idx, buffer_count + 1);
        pthread_mutex_lock(&writer_mutex);
        pthread_cond_signal(&writer_cond);
        pthread_mutex_unlock(&writer_mutex);
    }
    
    // Stop writer thread
    writer_running = 0;
    pthread_mutex_lock(&writer_mutex);
    pthread_cond_signal(&writer_cond);
    pthread_mutex_unlock(&writer_mutex);
    pthread_join(writer_thread, NULL);
    
    struct timespec end_time;
    clock_gettime(CLOCK_REALTIME, &end_time);
    
    fprintf(stderr, "\n=== Summary ===\n");
    fprintf(stderr, "End Time: %s", ctime(&end_time.tv_sec));
    double elapsed = (end_time.tv_sec - start_time.tv_sec) + 
                     (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
    fprintf(stderr, "Total Time: %.2f seconds\n", elapsed);
    fprintf(stderr, "Total Samples: %d\n", atomic_load(&total_samples_collected));
    fprintf(stderr, "Effective Sample Rate: %.1f Hz\n", atomic_load(&total_samples_collected) / elapsed);
    fprintf(stderr, "FIFO Overflows: %d\n", atomic_load(&fifo_overflows));
    fprintf(stderr, "Buffers Written: %d\n", total_buffers_written);
    
    if (atomic_load(&disk_write_error)) {
        fprintf(stderr, "\n*** WARNING: Disk write error occurred! ***\n");
        fprintf(stderr, "*** Some data may not have been saved. Check disk space. ***\n");
    }
    
    if (csv_file != stdout) {
        fclose(csv_file);
    }
    close(spi_fd);
    
    if (output_file) {
        fprintf(stderr, "Data saved to %s\n", output_file);
    } else {
        fprintf(stderr, "Data streamed to stdout\n");
    }
    return atomic_load(&disk_write_error) ? 1 : 0;
}