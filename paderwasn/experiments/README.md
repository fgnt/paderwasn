# Experiments
This directory contains scripts to run experiments.

## Comparison of Geometry Calibration Methods
- To run the geometry calibration using iterative data set matching [1] run
the following command:
```bash 
$ python -m paderwasn.experiments.calibration_method with 'iter_dsm'
``` 
- To run the geometry calibration using iterative data set matching [1] without 
weighting and fitness selection run the following command:
```bash 
$ python -m paderwasn.experiments.calibration_method with 'iter_dsm_simple'
``` 
- To run the GARDE-algorithm [2] run the following command:
```bash 
$ python -m paderwasn.experiments.calibration_method with 'garde'
``` 

## References
[1] Gburrek, T., Schmalenstroeer, J., Haeb-Umbach, R.: Geometry Calibration in
Wireless Acoustic Sensor Networks Utilizing DoA and Distance Information, In:
Sub. to EURASIP Journal on Audio, Speech, and Music Processing 

[2] Gburrek, T., Schmalenstroeer, J., Haeb-Umbach, R.: Iterative Geometry
Calibration from Distance Estimates for Wireless Acoustic Sensor Networks. In:
Sub. to Proc. IEEEInternational Conference on Acoustics, Speech and Signal
Processing(ICASSP) (2021). arXiv:2012.06142

