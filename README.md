# paderwasn
Paderwasn is a collection of methods for acoustic signal processing in wireless acoustic sensor networks (WASNs).

This repository is currently under construction.

## Installation
Install requirements:
```bash
$ pip install --user git+https://github.com/fgnt/lazy_dataset.git@ce8a833221580242e69d43e62361adca02478f79
$ pip install --user git+https://github.com/fgnt/paderbox.git@7fed5b44be2effcedb7a26778ada6c5668b1d6bd
```

Clone the repository:
```bash
$ git clone https://github.com/fgnt/paderwasn.git
```

Install package:
```bash
$ pip install --user -e paderwasn
```

## Content
* Geometry calibration:
    + Algorithms can be found in `paderwasn/geometry_calibration`.
        + Geometry calibration using iterative data set matching [1]
        + GARDE-algorithm [2]
    + See `paderwasn/experiments` to run experiments.
* Signal synchronization:
    + Source code for signal synchronization in the presence of time-varying sampling rate offsets and source position changes [3] will be available soon!
   
## References
[1] Gburrek, T., Schmalenstroeer, J., Haeb-Umbach, R.: Geometry Calibration in
Wireless Acoustic Sensor Networks Utilizing DoA and Distance Information, In:
Sub. to EURASIP Journal on Audio, Speech, and Music Processing 

[2] Gburrek, T., Schmalenstroeer, J., Haeb-Umbach, R.: Iterative Geometry
Calibration from Distance Estimates for Wireless Acoustic Sensor Networks. In:
Accepted for Proc. IEEE International Conference on Acoustics, Speech and Signal
Processing(ICASSP) (2021). arXiv:2012.06142

[3] Gburrek, T., Schmalenstroeer, J., Haeb-Umbach, R.: On Synchronization of
Wireless Acoustic Sensor Networks in the Presence of Time-Varying Sampling Rate
Offsets and Speaker Changes. In: Submitted to IEEE International Conference on
Acoustics, Speech and Signal Processing(ICASSP) (2022)
