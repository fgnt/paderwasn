"""
Example call:
python -m paderwasn.experiments.sto_estimation with 'scenario="Scenario-32'
"""
import numpy as np
from paderbox.array import segment_axis
from paderbox.io import load_audio
from sacred import Experiment

from paderwasn.databases.synchronization.database import AsyncWASN
from paderwasn.synchronization.sro_estimation import OnlineWACD, DynamicWACD
from paderwasn.synchronization.sync import coarse_sync, compensate_sro
from paderwasn.synchronization.sto_estimation import est_sto
from paderwasn.synchronization.time_shift_estimation import est_time_shift
from paderwasn.synchronization.utils import VoiceActivityDetector


ex = Experiment('Evaluate STO estimation')


@ex.config
def config():
    db_json = None
    msg = 'You have to specify the file path for the database json.'
    assert db_json is not None, msg
    scenario = 'Scenario-3'
    ref_node_id = 0
    vad_threshold = 3 * 9.774e-5 ** 2 * 1024
    activity_threshold = .75 * 16384


def get_distances(example):
    src_diary = example['src_diary']
    node_pos = \
        np.asarray([pos for pos in example['node__position'].values()])
    src_pos = np.zeros(((src_diary[-1]['offset']), 3))
    for src in src_diary:
        onset = src['onset']
        offset = src['offset']
        src_pos[onset:offset] = np.asarray(src['source_position'])
    dists = np.linalg.norm(src_pos[:, None] - node_pos[None], axis=-1)
    for src_id in range(1, len(src_diary)):
        offset = src_diary[src_id-1]['offset']
        onset = src_diary[src_id]['onset']
        dists[offset:onset] = np.random.uniform(.3, 6.)
    return dists


@ex.automain
def eval_estimator(db_json,
                   scenario,
                   ref_node_id,
                   vad_threshold,
                   activity_threshold):
    msg = ('scenario must be "Scenario-1", "Scenario-2", '
           '"Scenario-3" or "Scenario-4"')
    scenarios = ['Scenario-1', 'Scenario-2', 'Scenario-3', 'Scenario-4']
    assert scenario in scenarios, msg

    if scenario == 'Scenario-1':
        db = AsyncWASN(db_json).get_data_set_scenario_1()
    elif scenario == 'Scenario-2':
        db = AsyncWASN(db_json).get_data_set_scenario_2()
    elif scenario == 'Scenario-3':
        db = AsyncWASN(db_json).get_data_set_scenario_3()
    elif scenario == 'Scenario-4':
        db = AsyncWASN(db_json).get_data_set_scenario_4()

    sro_estimator = DynamicWACD()
    voice_activity_detector = VoiceActivityDetector(vad_threshold)
    num_examples = 3 * len(db)
    errors = np.zeros(num_examples)
    for ex_id, example in enumerate(db):
        print(f'Process example {example["example_id"].split("_")[-1]}')
        all_dists = get_distances(example)
        ref_sig = load_audio(example['audio_path'][f'node_{ref_node_id}'])
        other_nodes = [i for i in range(4) if i != ref_node_id]
        for cnt, node_id in enumerate(other_nodes):
            sig = load_audio(example['audio_path'][f'node_{node_id}'])

            # Align the signals coarsely
            sig_sync, ref_sig_sync, offset = \
                coarse_sync(sig, ref_sig, len_sync=320000)

            # Estimate the sampling rate offset (SRO)
            activity_sig = voice_activity_detector(sig_sync)
            activity_ref_sig = voice_activity_detector(ref_sig_sync)
            sro_est = sro_estimator(
                sig_sync, ref_sig_sync, activity_sig, activity_ref_sig
            )

            # Compensate for the SRO
            sig_sync = compensate_sro(sig_sync, sro_est)
            ref_sig_sync = ref_sig_sync[:len(sig_sync)]

            # Estimate the time shifts and distances
            sig_shifts = est_time_shift(sig_sync, ref_sig_sync, 16384, 2048)
            if offset > 0:
                dists = all_dists[int(np.round(offset)):, node_id]
                dists_ref = all_dists[:, ref_node_id]
            else:
                dists = all_dists[:, node_id]
                dists_ref = all_dists[int(np.round(-offset)):, ref_node_id]
            frame_ids = \
                8192 + np.asarray([i*2048 for i in range(len(sig_shifts))])
            dists = dists[frame_ids]
            dists_ref = dists_ref[frame_ids]

            # Discard estimates corresponding to periods in time
            # without source activity
            activity_ref_sig = voice_activity_detector(ref_sig_sync)
            activity_ref_sig = \
                (segment_axis(activity_ref_sig, 16384, 2048).sum(-1)
                 > activity_threshold)
            activity_sig = voice_activity_detector(sig_sync)
            activity_sig = (segment_axis(activity_sig, 16384, 2048).sum(-1)
                            > activity_threshold)
            activity_mask = np.logical_and(activity_sig, activity_ref_sig)
            sig_shifts = sig_shifts[activity_mask]
            dists = dists[activity_mask]
            dists_ref = dists_ref[activity_mask]

            # Estimate the sampling time offsett (STO)
            sto_est = est_sto(sig_shifts, dists, dists_ref) - offset

            # Calculate the estimation error
            sto = (example['sto'][f'node_{node_id}']
                   - example['sto'][f'node_{ref_node_id}'])
            errors[3*ex_id+cnt] = sto - sto_est
            print(f'node {node_id}: error = '
                  f'{np.round(errors[3*ex_id+cnt], 2)} samples')
    print(f'\nRMSE = {np.round(np.sqrt(np.mean(errors**2)), 2)} samples')
