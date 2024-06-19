"""
Perform source separation on LibriWASN.  Firstly, time-frequency masks
are estimated using a complex Angular Central Gaussian Mixture Model (cACGMM).
Afterwards, the speakers' signals are extracted using the joint sampling rate offset synchronization and source extraction via
beamforming approach as proposed in [Gburrek23]

@inproceedings{Gburrek23,
   author={Gburrek, Tobias and Schmalenstroeer, Joerg
           and Haeb-Umbach, Reinhold},
   booktitle = {31st European Signal Processing Conference (EUSIPCO)},
   pages = {1--5},
   title = {{On the Integration of Sampling Rate Synchronization and
             Acoustic Beamforming}},
   year = {2023},
}

Example calls:
python -m paderwasn.experiments.source_separation with single data_set="libriwasn200" storage_dir=/path/to/storage_diretory db_json=/path/to/libriwasn.json
python -m paderwasn.experiments.source_separation with extended data_set="libriwasn800" storage_dir=/path/to/storage_diretory db_json=/path/to/libriwasn.json
python -m paderwasn.experiments.source_separation with all data_set="libriwasn800" storage_dir=/path/to/storage_diretory db_json=/path/to/libriwasn.json


If dlp_mpi is provided, achieve speedup by  starting <num_processes> simultaneously:
mpiexec -np <num_processes> python -m paderwasn.experiments.source_separation with extended data_set="libriwasn800" storage_dir=/path/to/storage_diretory db_json=/path/to/libriwasn.json

"""
from pathlib import Path

import dlp_mpi
from lazy_dataset.database import JsonDatabase
import paderbox as pb
from sacred import Experiment, SETTINGS
from libriwasn.io.audioread import load_signals
from libriwasn.mask_estimation.initialization import get_initialization
from libriwasn.mask_estimation.cacgmm import get_tf_masks
import numpy as np
from libriwasn.source_extraction.activity import (
    estimate_noise_class,
    estimate_activity
)

from paderwasn.source_extraction.sync_and_beamform import synchronizing_block_online_mvdr


SETTINGS.CONFIG.READ_ONLY_CONFIG = False
exp = Experiment('Separate sources')


@exp.config
def config():
    db_json = None
    storage_dir = None
    data_set = None
    devices_cacgmm = None
    devices_mvdr = None
    ref_device_sync = None
    single_ch_list = None

@exp.named_config
def single():
    devices_cacgmm = 'asnupb4'
    devices_mvdr = 'asnupb4'
    ref_device_sync = 'asnupb4'

@exp.named_config
def extended():
    devices_cacgmm = 'asnupb4'
    devices_mvdr = ['asnupb4', 'asnupb2', 'asnupb7']
    single_ch_list = [False, True, True]
    ref_device_sync = 'asnupb4'

@exp.named_config
def all():
    devices_cacgmm = 'asnupb4'
    devices_mvdr = None
    ref_device_sync = 'asnupb4'


def sep_sources_mvdr(sigs, masks, priors, mic_groups):
    noise_class = estimate_noise_class(priors)
    sig_segments = []
    segment_onsets = []
    activity = np.asarray(
        [estimate_activity(priors[i]) for i in range(len(masks))])
    for target_spk in range(len(masks)):
        if target_spk == noise_class:
            continue
        sig_segments_spk, _, segment_onsets_spk = synchronizing_block_online_mvdr(
            target_spk, sigs, masks, activity, mic_groups,
            noise_class=noise_class, multi_ch_sro_est=True, return_onset=True)
        sig_segments.append(sig_segments_spk)
        segment_onsets.append(segment_onsets_spk)
    return sig_segments, segment_onsets


def get_mic_groups(sigs):
    if not type(sigs) == list:
        mic_groups = [list(np.arange(len(sigs)))]
    else:
        mic_groups = []
        cnt = 0
        for _sig in sigs:
            if _sig.ndim == 2:
                mic_groups.append(list(np.arange(_sig.shape[0]) + cnt))
                cnt += _sig.shape[0]
            else:
                mic_groups.append([cnt])
                cnt += 1
    return mic_groups


@exp.automain
def separate_sources(
        db_json, storage_dir, data_set, devices_cacgmm,
        devices_mvdr, ref_device_sync, single_ch_list
):
    msg = 'You have to specify, where your LibriWASN database-json is stored.'
    assert db_json is not None, msg
    storage_dir = Path(storage_dir).absolute()
    segment_json = storage_dir / 'per_utt.json'
    ds = JsonDatabase(db_json)
    ds = ds.get_dataset(data_set)

    enhanced_segments = {}
    for example in dlp_mpi.split_managed(ds, allow_single_worker=True):
        ex_id = example['example_id']
        audio_root = \
            storage_dir / example['overlap_condition'] / example["example_id"]

        # estimate time frequency masks
        if isinstance(devices_cacgmm, str):
            single_ch = False
        else:
            single_ch = True

        sigs, devices = load_signals(
            example, devices=devices_cacgmm, single_ch=single_ch,
            ref_device=ref_device_sync, return_devices=True,
        )
        assert len(devices) == 1
        y = pb.transform.stft(sigs)

        mic_groups = get_mic_groups(sigs)

        mm_init, mm_guide = get_initialization(y)
        masks, priors = get_tf_masks(y, mm_init, mm_guide)

        # separate sources
        if devices_cacgmm != devices_mvdr or sigs is None:
            if isinstance(devices_mvdr, str):
                single_ch = False
            else:
                single_ch = True
                
            if single_ch_list is None:
                single_ch_list = single_ch

            sigs, devices = load_signals(
                example, devices=devices_mvdr, single_ch=single_ch_list,
                ref_device=ref_device_sync, return_devices=True, same_len=True
            )
        mic_groups = get_mic_groups(sigs)       

        separated_sigs, segment_onsets = sep_sources_mvdr(sigs, masks, priors, mic_groups)

        for spk_id in range(len(separated_sigs)):
            path_sep_sigs_target = audio_root / str(spk_id)
            sigs_spk = separated_sigs[spk_id]
            onsets_spk = segment_onsets[spk_id]
            for idx, sig in enumerate(sigs_spk):
                if not path_sep_sigs_target.exists():
                    path_sep_sigs_target.mkdir(parents=True)
                segment_id = f'{ex_id}_{spk_id}_{idx}'
                audio_path = \
                    path_sep_sigs_target / f'enhanced{spk_id}_{idx}.wav'
                short_id = \
                    f'{example["overlap_condition"]}_{example["session"]}'
                enhanced_segments[segment_id] = {
                    "audio_path": audio_path,
                    "dataset": "eval",
                    "short_id": short_id,
                    "speaker_id": str(spk_id),
                    "start_sample": onsets_spk[idx],
                    "stop_sample": onsets_spk[idx] + len(sig)
                }
                pb.io.dump_audio(sig, audio_path)
        # reduce memory consumption
        del sigs, y, mm_init, mm_guide, masks, priors, separated_sigs

    all_enh_segments = dlp_mpi.gather(enhanced_segments, root=dlp_mpi.MASTER)
    if dlp_mpi.IS_MASTER:
        all_segments_flattened = {}
        for seg in all_enh_segments:
            all_segments_flattened.update(seg)
        pb.io.dump_json(
            all_segments_flattened, segment_json
        )
        print(f'Wrote: {segment_json}', flush=True)
