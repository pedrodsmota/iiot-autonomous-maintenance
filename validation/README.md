# validation/ - validation of the article

The system was evaluated at four levels. The model evaluation is in
[`../results/`](../results/); the other three are below.

## One hour of operation (`control_acquisition/`)

Production cycles, MQTT communication, and continuity of the sensor streams, over one hour of
continuous operation of the line. `cycle_duration.py`, `mqtt_round_trip.py`, `record_arrivals.c`,
and `analyze_streams.py` are the tools used. The script that measured the coordination latency
(mean 12.4 ms, maximum 114 ms) was not preserved.

## Trigger of the replacement part (`trigger_test/`)

90 trials, in three batches of 30. An MQTT client published the print order to the broker of the
AdM node, in a different laboratory, over the wireless network of the university. `PRINT_QUEUE`
was bypassed, and an additional MQTT publisher block on the AdM node returned an acknowledgment
once OctoPrint had accepted the print-start request. The original script was not preserved;
`print_validation.py` reproduces its procedure and output format. `results.csv` has the 90 trials.

| Arm | Trials | Successful | Mean latency (ms) | sigma (ms) |
|---|---|---|---|---|
| #1 | 30 | 30 | 350 | 98 |
| #2 | 30 | 30 | 339 | 83 |
| #3 | 30 | 30 | 388 | 109 |

## End-to-end trials (procedure; no script)

Three trials, one per arm, conducted sequentially because a single printer was available, on newly
degraded grippers rather than on the data used to train and test the models:

1. The online estimation runs on live vibration data until the estimated HI reaches the threshold of
   the arm, without forcing it.
2. The resulting print order travels through the distributed system to the AdM node, and the print
   starts without human intervention in the decision.
3. The response time is measured locally on the AdM node, from the reception of the order to the
   response of the OctoPrint API, so it excludes the network transmission.
4. The print is timed, and the operation continues until the gripper fails completely. Post-processing
   and replacing the part were not performed during the trials; an allowance of about 10 minutes is
   added to the print duration.

| Arm | Threshold crossing | AdM response time | Production and replacement | Complete failure | Advance notice | Margin |
|---|---|---|---|---|---|---|
| #1 | 18 h 39 min | 336 ms | 3 h 27 min | 22 h 14 min | 3 h 35 min | 8 min |
| #2 | 18 h 47 min | 314 ms | 3 h 36 min | 22 h 25 min | 3 h 38 min | 2 min |
| #3 | 18 h 43 min | 258 ms | 3 h 34 min | 22 h 19 min | 3 h 36 min | 2 min |
