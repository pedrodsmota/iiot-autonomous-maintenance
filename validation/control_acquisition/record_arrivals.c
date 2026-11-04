/*
 * One-hour control and acquisition test - logger of the sensor stream on a PdM node
 * (dissertation, Section 6.2).
 *
 * Reads the stream line by line from stdin and writes, for every line, the time at
 * which it arrived (Unix time, s). With -s it also writes the first number of the
 * line, i.e. the sensor time stamp (ADXL345, "t,x,y,z") or the sample index
 * (MPU9250), which analyze_streams.py uses to detect lost samples.
 *
 * Build:  gcc -O2 -o record_arrivals record_arrivals.c
 * Usage:
 *   Arms 1 and 3 (MQTT):  mosquitto_sub -t <stream topic> | ./record_arrivals -s > arm1_stream.csv
 *   Arm 2 (serial):       timeout 3600 cat /dev/ttyACM0 | ./record_arrivals > arm2_stream.csv
 * The output is flushed after every line, so the log survives the end of the test.
 */
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

int main(int argc, char *argv[]) {
    char line[256];
    struct timeval tv;
    double sensor_time;
    int with_sensor_time = (argc > 1 && strcmp(argv[1], "-s") == 0);

    printf(with_sensor_time ? "local_arrival_time,sensor_time\n" : "local_arrival_time\n");
    while (fgets(line, sizeof(line), stdin)) {
        gettimeofday(&tv, NULL);   /* as soon as the line is read */
        double local_time = (double)tv.tv_sec + (double)tv.tv_usec / 1e6;
        if (!with_sensor_time)
            printf("%.6f\n", local_time);
        else if (sscanf(line, "%lf", &sensor_time) == 1)
            printf("%.6f,%.6f\n", local_time, sensor_time);
        fflush(stdout);
    }
    return 0;
}
