# Experiments
This directory contains scripts to run experiments.

## Comparison of Geometry Calibration Methods
- To run the geometry calibration using iterative data set matching [1] run
the following command:
```bash 
$ python -m paderwasn.experiments.calibration_methods with 'iter_dsm'
``` 
- To run the geometry calibration using iterative data set matching [1] without 
weighting and fitness selection run the following command:
```bash 
$ python -m paderwasn.experiments.calibration_methods with 'iter_dsm_simple'
``` 
- To run the GARDE-algorithm [2] run the following command:
```bash 
$ python -m paderwasn.experiments.calibration_methods with 'garde'
``` 
## Signal Synchronization
All experiments are done on the ansynchronous WASN database used in [3] (see [3] for more details):

| Scenario | time-varying SRO | Multiple Source Positions | Speech Pauses |
| :-----------: | :-----------: |  :-----------: |  :-----------: |
| Scenario-1  | | | |
| Scenario-2  | X | | |
| Scenario-3  | X | X | X |
| Scenario-4  | X | X | |

To select Scenario-x, with x from {1, 2, 3, 4}, for the experiments append
``` 'scenario="Scenario-x"' ``` to the end of the commands mentioned below.

### Comparison of Sampling Rate Offset Estimation Methods
- To run the DWACD algorithm [3] run the following command:
```bash 
$ python -m paderwasn.experiments.sro_estimation_methods with 'method="DWACD"' 'db_json="/PATH/TO/ASYC_WASN_DB_JSON/"'
``` 
- To run the online WACD algorithm [4] run the following command:
```bash 
$ python -m paderwasn.experiments.sro_estimation_methods with 'method="online WACD"' 'db_json="/PATH/TO/ASYC_WASN_DB_JSON/"'
``` 

### Sampling Time Offset Estimation
To run the  sampling time offset estimator proposed in  [1] run the following command:
```bash 
$ python -m paderwasn.experiments.sto_estimation with 'db_json="/PATH/TO/ASYC_WASN_DB_JSON/"'
``` 

## References
[1] Gburrek, T., Schmalenstroeer, J., Haeb-Umbach, R.: "Geometry Calibration in
Wireless Acoustic Sensor Networks Utilizing DoA and Distance Information", 
EURASIP Journal on Audio, Speech, and Music Processing, vol. 2021, no. 1,
pp. 1–17, 2021.

[2] Gburrek, T., Schmalenstroeer, J., Haeb-Umbach, R.: "Iterative Geometry
Calibration from Distance Estimates for Wireless Acoustic Sensor Networks". in
Proc. IEEE International Conference on Acoustics, Speech and Signal Processing
(ICASSP), 2021, pp. 741-745.

[3] Gburrek, T., Schmalenstroeer, J., Haeb-Umbach, R.: "On Synchronization of
Wireless Acoustic Sensor Networks in the Presence of Time-varying Sampling Rate
Offsets and Speaker Changes". Submitted to IEEE International Conference on
Acoustics, Speech and Signal Processing (ICASSP), 2022, arXiv preprint
arXiv:2110.12820.

[4] Chinaev, A., Enzner, G., Gburrek, T., Schmalenstroeer, J.:
“Online Estimation of Sampling Rate Offsets in Wireless Acoustic Sensor
Networks with Packet Loss,” in Proc. 29th European Signal Processing Conference
(EUSIPCO), 2021, pp. 1–5.
