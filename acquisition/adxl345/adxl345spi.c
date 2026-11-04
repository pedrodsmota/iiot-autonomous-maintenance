/*
 * adxl345spi (modified) - ADXL345 sampling over SPI on a Raspberry Pi.
 *
 * Based on adxl345spi by Ruslan Nagimov, https://github.com/nagimov/adxl345spi
 * (commit a05f03a8b97b07a4b1423cf9fb5372fdcf084add), licensed under the
 * GNU General Public License v3.0. This modified version is distributed under
 * the same license (see LICENSE in this folder).
 *
 * Changes with respect to the original (version 0.3 -> 0.4):
 *   - Long recordings: the original keeps every sample in RAM and writes the
 *     file at the end, which does not fit one-hour sessions on a Raspberry Pi 3.
 *     This version processes the raw SPI readings in chunks of about one second
 *     and writes the output continuously, through three alternating buffers and
 *     a writer thread pinned to another CPU core, so that sampling is never
 *     blocked by storage.
 *   - Graceful stop on SIGINT (the partial buffer is written before exiting).
 *   - Disk write errors (e.g. a full disk) stop the recording instead of
 *     silently losing data.
 * Used in the dissertation at 3200 Hz (-f 3200), Section 5.3.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <pigpio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdatomic.h>
#include <sched.h>
#include <signal.h>

#define DATA_FORMAT   0x31  // data format register address
#define DATA_FORMAT_B 0x0B  // data format bytes: +/- 16g range, 13-bit resolution (p. 26 of ADXL345 datasheet)
#define READ_BIT      0x80
#define MULTI_BIT     0x40
#define BW_RATE       0x2C
#define POWER_CTL     0x2D
#define DATAX0        0x32

const char codeVersion[3] = "0.4";  // code version number
const int timeDefault = 5;  // default duration of data stream, seconds
const int freqDefault = 5;  // default sampling rate of data stream, Hz
const int freqMax = 3200;  // maximal allowed cmdline arg sampling rate of data stream, Hz
const int speedSPI = 2000000;  // SPI communication speed, bps
const int freqMaxSPI = 100000;  // maximal possible communication sampling rate through SPI, Hz (assumption)
const int coldStartSamples = 2;  // number of samples to be read before outputting data to console (cold start delays)
const double coldStartDelay = 0.1;  // time delay between cold start reads
const double accConversion = 2 * 16.0 / 8192.0;  // +/- 16g range, 13-bit resolution
const double tStatusReport = 1;  // time period of status report if data read to file, seconds

// Buffer configuration for threaded file writing
// Each buffer holds 1 second of data at max rate
#define BUFFER_SAMPLES 4000
#define NUM_BUFFERS 3

typedef struct {
    double t[BUFFER_SAMPLES];
    double x[BUFFER_SAMPLES];
    double y[BUFFER_SAMPLES];
    double z[BUFFER_SAMPLES];
    int count;
    int buffer_id;
} SampleBuffer;

// Triple buffer system
SampleBuffer buffers[NUM_BUFFERS];
atomic_int buffer_write_idx = 0;
atomic_int buffer_fill_idx = 0;
atomic_int buffers_ready[NUM_BUFFERS];

// Writer thread control
atomic_int writer_running = 1;
atomic_int total_buffers_written = 0;
atomic_int disk_write_error = 0;
pthread_t writer_thread;
pthread_mutex_t writer_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t writer_cond = PTHREAD_COND_INITIALIZER;

// Global state
FILE *output_file = NULL;
volatile sig_atomic_t stop_requested = 0;

void signal_handler(int sig) {
    (void)sig;
    stop_requested = 1;
    printf("\nStop requested. Finishing up...\n");
}

void printUsage() {
    printf( "adxl345spi (version %s) \n"
            "License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>\n"
            "\n"
            "Usage: adxl345spi [OPTION]... \n"
            "Read data from ADXL345 accelerometer through SPI interface on Raspberry Pi.\n"
            "Online help, docs & bug reports: <https://github.com/nagimov/adxl345spi/>\n"
            "\n"
            "Mandatory arguments to long options are mandatory for short options too.\n"
            "  -s, --save FILE     save data to specified FILE (data printed to command-line\n"
            "                      output, if not specified)\n"
            "  -t, --time TIME     set the duration of data stream to TIME seconds\n"
            "                      (default: %i seconds) [integer]\n"
            "  -f, --freq FREQ     set the sampling rate of data stream to FREQ samples per\n"
            "                      second, 1 <= FREQ <= %i (default: %i Hz) [integer]\n"
            "\n"
            "Data is streamed in comma separated format, e. g.:\n"
            "  time,     x,     y,     z\n"
            "   0.0,  10.0,   0.0, -10.0\n"
            "   1.0,   5.0,  -5.0,  10.0\n"
            "   ...,   ...,   ...,   ...\n"
            "  time shows seconds elapsed since the first reading;\n"
            "  x, y and z show acceleration along x, y and z axis in fractions of <g>.\n"
            "\n"
            "Exit status:\n"
            "  0  if OK\n"
            "  1  if error occurred during data reading or wrong cmdline arguments.\n"
            "", codeVersion, timeDefault, freqMax, freqDefault);
}

int readBytes(int handle, char *data, int count) {
    data[0] |= READ_BIT;
    if (count > 1) data[0] |= MULTI_BIT;
    return spiXfer(handle, data, data, count);
}

int writeBytes(int handle, char *data, int count) {
    if (count > 1) data[0] |= MULTI_BIT;
    return spiWrite(handle, data, count);
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
            if (fprintf(output_file, "%.5f, %.5f, %.5f, %.5f\n", 
                    buf->t[i], buf->x[i], buf->y[i], buf->z[i]) < 0) {
                write_error = 1;
            }
        }
        
        if (fflush(output_file) != 0 || write_error) {
            fprintf(stderr, "\n*** DISK WRITE ERROR (disk full?) ***\n");
            fprintf(stderr, "*** Stopping to preserve data integrity ***\n");
            atomic_store(&disk_write_error, 1);
            stop_requested = 1;
            break;
        }
        
        // Mark buffer as available
        atomic_store(&buffers_ready[idx], 0);
        atomic_fetch_add(&buffer_write_idx, 1);
        total_buffers_written++;
        
        printf("  [Writer] Buffer %d written (%d samples)\n", buf->buffer_id, buf->count);
    }
    
    return NULL;
}

int main(int argc, char *argv[]) {
    int i;

    // handling command-line arguments

    int bSave = 0;
    char vSave[256] = "";
    double vTime = timeDefault;
    double vFreq = freqDefault;
    for (i = 1; i < argc; i++) {  // skip argv[0] (program name)
        if ((strcmp(argv[i], "-s") == 0) || (strcmp(argv[i], "--save") == 0)) {
            bSave = 1;
            if (i + 1 <= argc - 1) {  // make sure there are enough arguments in argv
                i++;
                strcpy(vSave, argv[i]);
            }
            else {
                printUsage();
                return 1;
            }
        }
        else if ((strcmp(argv[i], "-t") == 0) || (strcmp(argv[i], "--time") == 0)) {
            if (i + 1 <= argc - 1) {
                i++;
                vTime = atoi(argv[i]);
            }
            else {
                printUsage();
                return 1;
            }
        }
        else if ((strcmp(argv[i], "-f") == 0) || (strcmp(argv[i], "--freq") == 0)) {
            if (i + 1 <= argc - 1) {
                i++;
                vFreq = atoi(argv[i]);
                if ((vFreq < 1) || (vFreq > 3200)) {
                    printf("Wrong sampling rate specified!\n\n");
                    printUsage();
                    return 1;
                }
            }
            else {
                printUsage();
                return 1;
            }
        }
        else {
            printUsage();
            return 1;
        }
    }

    // reading sensor data

    // SPI sensor setup
    int samples = vFreq * vTime;
    // Note: samplesMaxSPI is only used for console output mode (not file mode)
    int success = 1;
    int h, bytes;
    char data[7];
    int16_t x, y, z;
    double tStart, tDuration, t;
    if (gpioInitialise() < 0) {
        printf("Failed to initialize GPIO!");
        return 1;
    }
    h = spiOpen(0, speedSPI, 3);
    data[0] = BW_RATE;
    data[1] = 0x0F;
    writeBytes(h, data, 2);
    data[0] = DATA_FORMAT;
    data[1] = DATA_FORMAT_B;
    writeBytes(h, data, 2);
    data[0] = POWER_CTL;
    data[1] = 0x08;
    writeBytes(h, data, 2);

    double delay = 1.0 / vFreq;  // delay between reads in seconds

    // depending from the output mode (print to cmdline / save to file) data is read in different ways

    // for cmdline output, data is read directly to the screen with a sampling rate which is *approximately* equal...
    // but always less than the specified value, since reading takes some time

    if (bSave == 0) {
        // fake reads to eliminate cold start timing issues (~0.01 s shift of sampling time after the first reading)
        for (i = 0; i < coldStartSamples; i++) {
            data[0] = DATAX0;
            bytes = readBytes(h, data, 7);
            if (bytes != 7) {
                success = 0;
            }
            time_sleep(coldStartDelay);
        }
        // real reads happen here
        tStart = time_time();
        for (i = 0; i < samples; i++) {
            data[0] = DATAX0;
            bytes = readBytes(h, data, 7);
            if (bytes == 7) {
                x = (data[2]<<8)|data[1];
                y = (data[4]<<8)|data[3];
                z = (data[6]<<8)|data[5];
                t = time_time() - tStart;
                printf("%.5f,%.5f,%.5f,%.5f\n",
                       t, x * accConversion, y * accConversion, z * accConversion);
                }
            else {
                success = 0;
            }
            time_sleep(delay);  // pigpio sleep is accurate enough for console output, not necessary to use nanosleep
        }
        gpioTerminate();
        tDuration = time_time() - tStart;  // need to update current time to give a closer estimate of sampling rate
        printf("%d samples read in %.2f seconds with sampling rate %.1f Hz\n", samples, tDuration, samples/tDuration);
        if (success == 0) {
            printf("Error occurred!");
            return 1;
        }
    }

    // for the file output, data is read with a maximal possible sampling rate (around 30,000 Hz)...
    // and then accurately rescaled to *exactly* match the specified sampling rate...
    // therefore, saved data can be easily analyzed (e. g. with fft)
    // 
    // MODIFIED: Now uses triple-buffering and a writer thread to avoid RAM filling up.
    // Data is resampled in chunks and written continuously to disk.
    else {
        // Setup signal handler for graceful shutdown
        signal(SIGINT, signal_handler);
        signal(SIGTERM, signal_handler);
        
        // Try to pin main thread to CPU 0
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(0, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
        
        // Initialize buffers
        for (int b = 0; b < NUM_BUFFERS; b++) {
            buffers[b].count = 0;
            buffers[b].buffer_id = -1;
            atomic_store(&buffers_ready[b], 0);
        }
        
        // Open output file and start writer thread
        output_file = fopen(vSave, "w");
        if (!output_file) {
            printf("Failed to open output file: %s\n", vSave);
            gpioTerminate();
            return 1;
        }
        fprintf(output_file, "time, x, y, z\n");
        fflush(output_file);
        
        if (pthread_create(&writer_thread, NULL, writer_thread_func, NULL) != 0) {
            printf("Failed to create writer thread!\n");
            fclose(output_file);
            gpioTerminate();
            return 1;
        }

        // Chunk-based raw data buffers (process in chunks to limit RAM usage)
        // Each chunk is ~1 second of raw data at max SPI rate
        const int CHUNK_RAW_SIZE = freqMaxSPI;  // 1 second of raw samples
        double *rt, *rx, *ry, *rz;
        rt = malloc(CHUNK_RAW_SIZE * sizeof(double));
        rx = malloc(CHUNK_RAW_SIZE * sizeof(double));
        ry = malloc(CHUNK_RAW_SIZE * sizeof(double));
        rz = malloc(CHUNK_RAW_SIZE * sizeof(double));
        
        if (!rt || !rx || !ry || !rz) {
            printf("Failed to allocate memory!\n");
            gpioTerminate();
            return 1;
        }

        printf("Reading with sampling rate %.1f Hz using chunked buffering...\n", vFreq);
        printf("Press Ctrl+C to stop early.\n");
        
        int statusReportedTimes = 0;
        double tLeft;
        int current_buffer_idx = 0;
        int buffer_count = 0;
        SampleBuffer *current_buffer = &buffers[0];
        current_buffer->count = 0;
        current_buffer->buffer_id = 0;
        
        int total_output_samples = 0;
        double next_output_time = 0.0;  // Next time point we need a sample for
        
        tStart = time_time();
        int chunk_samples = 0;
        
        while (!stop_requested && total_output_samples < samples) {
            // Read one raw sample
            data[0] = DATAX0;
            bytes = readBytes(h, data, 7);
            if (bytes == 7) {
                x = (data[2]<<8)|data[1];
                y = (data[4]<<8)|data[3];
                z = (data[6]<<8)|data[5];
                t = time_time() - tStart;
                
                rx[chunk_samples] = x * accConversion;
                ry[chunk_samples] = y * accConversion;
                rz[chunk_samples] = z * accConversion;
                rt[chunk_samples] = t;
                chunk_samples++;
                
                // Check if this sample (or a nearby one) should be output
                // We output when we have a sample at or past the next_output_time
                while (next_output_time <= t && total_output_samples < samples) {
                    // Find the closest sample in recent chunk for this output time
                    int best_idx = chunk_samples - 1;
                    double best_error = fabs(rt[best_idx] - next_output_time);
                    
                    // Look back a few samples for a better match
                    for (int k = chunk_samples - 2; k >= 0 && k >= chunk_samples - 10; k--) {
                        double err = fabs(rt[k] - next_output_time);
                        if (err < best_error) {
                            best_error = err;
                            best_idx = k;
                        }
                    }
                    
                    // Add to output buffer
                    int idx = current_buffer->count;
                    current_buffer->t[idx] = next_output_time;
                    current_buffer->x[idx] = rx[best_idx];
                    current_buffer->y[idx] = ry[best_idx];
                    current_buffer->z[idx] = rz[best_idx];
                    current_buffer->count++;
                    total_output_samples++;
                    next_output_time = (double)total_output_samples * delay;
                    
                    // Buffer full? Swap to next
                    if (current_buffer->count >= BUFFER_SAMPLES) {
                        atomic_store(&buffers_ready[current_buffer_idx], 1);
                        pthread_mutex_lock(&writer_mutex);
                        pthread_cond_signal(&writer_cond);
                        pthread_mutex_unlock(&writer_mutex);
                        
                        buffer_count++;
                        printf("Buffer %d filled (%d total output samples)\n", 
                               current_buffer->buffer_id, total_output_samples);
                        
                        current_buffer_idx = (current_buffer_idx + 1) % NUM_BUFFERS;
                        current_buffer = &buffers[current_buffer_idx];
                        
                        // Wait if next buffer is still being written
                        int wait_count = 0;
                        while (atomic_load(&buffers_ready[current_buffer_idx])) {
                            time_sleep(0.001);
                            wait_count++;
                            if (wait_count > 1000) {
                                fprintf(stderr, "ERROR: Writer thread too slow!\n");
                                break;
                            }
                        }
                        
                        current_buffer->count = 0;
                        current_buffer->buffer_id = buffer_count;
                        atomic_store(&buffer_fill_idx, buffer_count);
                    }
                }
                
                // Reset chunk buffer periodically to avoid overflow
                if (chunk_samples >= CHUNK_RAW_SIZE - 1) {
                    // Keep only the last few samples for continuity
                    int keep = 100;
                    memmove(rt, rt + chunk_samples - keep, keep * sizeof(double));
                    memmove(rx, rx + chunk_samples - keep, keep * sizeof(double));
                    memmove(ry, ry + chunk_samples - keep, keep * sizeof(double));
                    memmove(rz, rz + chunk_samples - keep, keep * sizeof(double));
                    chunk_samples = keep;
                }
            }
            else {
                gpioTerminate();
                printf("Error occurred during SPI read!");
                return 1;
            }
            
            tDuration = t;
            if (tDuration > tStatusReport * ((float)statusReportedTimes + 1.0)) {
                statusReportedTimes++;
                tLeft = vTime - tDuration;
                if (tLeft < 0) tLeft = 0.0;
                printf("%.2f seconds left... (%d samples collected)\n", tLeft, total_output_samples);
            }
            
            // Check for disk error from writer thread
            if (atomic_load(&disk_write_error)) {
                break;
            }
        }
        
        gpioTerminate();
        printf("\nStopping capture...\n");
        
        // Write last partial buffer
        if (current_buffer->count > 0 && !atomic_load(&disk_write_error)) {
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
        
        tDuration = time_time() - tStart;
        
        printf("\n=== Summary ===\n");
        printf("Total Time: %.2f seconds\n", tDuration);
        printf("Output Samples: %d (target: %d)\n", total_output_samples, samples);
        printf("Effective Output Rate: %.1f Hz\n", total_output_samples / tDuration);
        printf("Buffers Written: %d\n", total_buffers_written);
        
        if (atomic_load(&disk_write_error)) {
            printf("\n*** WARNING: Disk write error occurred! ***\n");
            printf("*** Some data may not have been saved. Check disk space. ***\n");
        }
        
        fclose(output_file);
        free(rt); free(rx); free(ry); free(rz);
    }

    printf("Done\n");
    return atomic_load(&disk_write_error) ? 1 : 0;
}
