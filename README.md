# paderwasn
Paderwasn is a collection of methods for acoustic signal processing in wireless acoustic sensor networks (WASNs).

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
* Algorithms:
    + [Geometry calibration](paderwasn/geometry_calibration):
        + Geometry calibration using iterative data set matching [1]
        + GARDE-algorithm [2]
    + [Signal synchronization](paderwasn/synchronization):
        + Sampling rate offset (SRO) estimation:
            + Dynamic weighted average coherence drift (WACD) [3]
            + Onlne WACD [4]
        + Sampling time offset (STO) estimation [3]
        + Resampling to compensate for an SRO
        + Simulation of a (time-varying) SRO [3]
* Databases:
    + [Geometry calibration observations](paderwasn/databases/geometry_calibration): Collection of direction-of-arrival (DoA) and source-node distance
        estimates used for geometry calibration in [1]
    + [Asynchronous WASN database](paderwasn/databases/synchronization): Database of simulated audio signals which were recorded by an
        asynchronous WASN. This database corresponds to the database (after
        minimal adjustments) used in [3] for evaluation of signal
        synchronization algorithms. 
* Experiments using the provided algorithms:
    + [Comparision of geometry calibration methods](paderwasn/experiments/calibration_methods.py)
    + [Comparision of SRO methods](paderwasn/experiments/sro_estimation_methods.py)
    + [STO estimation](paderwasn/experiments/sto_estimation.py)
   
## Asynchronous WASN database 
A description how to download and use the asynchronous WASN database [3] is
going to be added soon.


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

## Citation
If you are using the code or one of the provided databases please cite the
corresponding paper:
 ```
@article{gburrek2021geometry,
	title={Geometry calibration in wireless acoustic sensor networks utilizing DoA and distance information},
	author={Gburrek, Tobias and Schmalenstroeer, Joerg and Haeb-Umbach, Reinhold},
	journal={EURASIP Journal on Audio, Speech, and Music Processing},
	volume={2021},
	number={1},
	pages={1--17},
	year={2021},
	publisher={Springer}
}
```
 ```
@inproceedings{gburrek2021synchronization, 
    author={Gburrek, Tobias and Schmalenstroeer, Joerg and Haeb-Umbach, Reinhold}, 
    booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
    title={Iterative Geometry Calibration from Distance Estimates for Wireless Acoustic Sensor Networks},
    year={2021},
    pages={741-745},
    doi={10.1109/ICASSP39728.2021.9413831}
}
```
 ```
@misc{gburrek2021synchronization,
      title={On Synchronization of Wireless Acoustic Sensor Networks in the Presence of Time-varying Sampling Rate Offsets and Speaker Changes}, 
      author={Gburrek, Tobias and Schmalenstroeer, Joerg and Haeb-Umbach, Reinhold},
      year={2021},
      eprint={2110.12820},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```
 ```
@inproceedings{Chinaev2021,
	author = {Chinaev, Aleksej and Enzner, Gerald and Gburrek, Tobias and Schmalenstroeer, Joerg},
	booktitle = {29th European Signal Processing Conference (EUSIPCO)},
	pages = {1--5},
	title = {{Online Estimation of Sampling Rate Offsets in Wireless Acoustic Sensor Networks with Packet Loss}},
	year = {2021},
}
 ```

## Acknowledgment
Funded by the Deutsche Forschungsgemeinschaft (DFG, German Research
Foundation) - Project 282835863 ([Deutsche Forschungsgemeinschaft - 
DFG-FOR 2457](https://www.uni-paderborn.de/asn/)).

![img](https://www.uni-paderborn.de/fileadmin/_processed_/9/2/csm_ASNLogo_c443ce161b.png)
