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

| Scenario | Time-varying SRO | Multiple Source Positions | Speech Pauses |
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

## Source separation
This example shows the transcription performance of the integrated sampling rate synchronization and acoustic beamforming approach proposed in [5]
on [LibriWASN](https://github.com/fgnt/libriwasn/blob/main/libriwasn/reference_system/separate_sources.py]) [6] with varying microphone constellations
and compares it with the LibriWASN [reference system](https://github.com/fgnt/libriwasn/blob/main/libriwasn/reference_system/separate_sources.py) which uses the dynamic weighted average coherence
drift (DWACD) method for SRO estimation from [3].
Firstly, time-frequency masks are estimated using a complex Angular Central Gaussian Mixture Model (cACGMM).
Afterwards, the speakers' signals are extracted using the joint sampling rate offset synchronization and source extraction via beamforming.


To run the source separation and extract speaker utterances run the following command selecting `<setup>` from `{"single", "extended", "all"}`:
```bash 
$ python -m paderwasn.experiments.source_separation with <setup> data_set="libriwasn200" storage_dir=/path/to/storage_diretory db_json=/path/to/libriwasn.json
```
A speedup can be achieved by  starting `<num_processes>` simultaneously:
```bash
mpiexec -np <num_processes> python -m paderwasn.experiments.source_separation with <setup> data_set="libriwasn800" storage_dir=/path/to/storage_diretory db_json=/path/to/libriwasn.json
``` 
For the generation of transcriptions and WERs, follow the instructions from the second step of "reference system" in  the [README](https://github.com/fgnt/libriwasn?tab=readme-ov-file#reference-sytem) of LibriWASN.

### Results
#### Setup
- "asnupb4" (6 channels) is used to for the estimation of time-frequency masks using all channels and as reference device for sampling rate synchronization in all following setups.
- Setup "single" uses these 6 channels for the beamforming referring to a single-array scenario identical to "sys2" of the LibriWASN reference system.
- Setup "extended" uses these 6 channels and additional the first channel of "asnupb2" and "asnupb7" for the beamforming which constitutes a scenario of a single-array with two additional asynchronous microphones with overall 8 channels from 3 different devices. This corresponds to the simulated setup "Array + 2 async. mics" used in [5]. 
- Setup "all" uses the first channel of all devices present in LibriWASN recordings for the beamforming which is analog to "sys3" of the LibriWASN reference system with overall 9 channels from 9 different devices.
- Thy column "Sync." refers to the sampling rate offset estimation method used whereas "DWACD" refers to the LibriWASN reference system whereas "SCM" refers to the  integrated sampling rate synchronization and acoustic beamforming approach of this example. 

#### cpWER / % for LibriWASN200  
|  Setup   | Sync. |  0L  | 0S | OV10 | OV20 | OV30 | OV40 | Avg. |
|:--------:|:-----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|  single  | DWACD | 3.12 | 3.22 | 4.58 | 5.22 | 5.00 | 4.51 | 4.38 |
|  single  |  SCM  | 3.22 | 2.96 | 4.31 | 4.48 | 4.09 | 3.38 | 3.78 |
| extended | DWACD | 2.96 | 2.87 | 4.17 | 4.34 | 3.75 | 3.22 | 3.59 |
| extended |  SCM  | 3.14 | 2.85 | 4.31 | 4.59 | 4.04 | 3.45 | 3.77 |
|   all    | DWACD | 3.16 | 3.06 | 4.40 | 4.38 | 3.79 | 3.36 | 3.72 |
|   all    |  SCM  | 3.09 | 2.91 | 4.21 | 4.22 | 3.33 | 3.04 | 3.48 |

#### cpWER / % for LibriWASN800 
|  Setup   | Sync. |  0L  |  0S  | OV10 | OV20 | OV30 | OV40 | Avg. |
|:--------:|:-----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|  single  | DWACD | 3.89 | 3.94 | 5.23 | 7.13 | 7.15 | 6.84 | 5.89 |
|  single  |  SCM  | 3.98 | 3.63 | 4.38 | 5.72 | 5.85 | 5.31 | 4.92 |
| extended | DWACD | 3.92 | 3.49 | 4.19 | 5.58 | 5.50 | 5.04 | 4.71 |
| extended |  SCM  | 3.99 | 3.63 | 4.39 | 5.81 | 5.61 | 5.26 | 4.88 |
|   all    | DWACD | 3.92 | 3.55 | 4.14 | 5.37 | 5.01 | 4.53 | 4.48 |
|   all    |  SCM  | 3.70 | 3.37 | 3.86 | 4.80 | 4.41 | 3.86 | 4.03 |


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

[5]  Gburrek, T., Schmalenstroeer, J., Haeb-Umbach, R.:
“On the Integration of Sampling Rate Synchronization and Acoustic Beamforming”. in Proc. European Signal Processing Conference
(EUSIPCO), 2023.

[6] Schmalenstroeer, J., Gburrek, T., Haeb-Umbach, R.:
"LibriWASN: A Data Set for Meeting Separation, Diarization, and Recognition with Asynchronous Recording Devices",
in Proc. 15th ITG conference on Speech Communication, 2023, pp.86 - 91
