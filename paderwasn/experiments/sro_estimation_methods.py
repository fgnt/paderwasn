"""
Example call:
python -m paderwasn.experiments.sro_estimation_methods with 'method="DWACD"'
"""
import numpy as np
from paderbox.io import load_audio
from sacred import Experiment

from paderwasn.databases.synchronization.database import AsnycWASN
from paderwasn.databases.synchronization.utils import load_binary
from paderwasn.synchronization.sro_estimation import OnlineWACD, DynamicWACD
from paderwasn.synchronization.sync import coarse_sync
from paderwasn.synchronization.utils import VoiceActivityDetector


ex = Experiment('Evaluate SRO Estimator')


@ex.config
def config():
    db_json = None
    msg = 'You have to specify the file path for the database json.'
    assert db_json is not None, msg
    method = 'DWACD'
    scenario = 'Scenario-1'
    ref_node_id = 0
    vad_threshold = 3 * 9.774e-5 ** 2 * 1024
    len_coarse_sync = 320000


@ex.automain
def eval_estimator(db_json,
                   method,
                   scenario,
                   ref_node_id,
                   vad_threshold,
                   len_coarse_sync):
    assert method in ['DWACD', 'online WACD'], \
        'method must be "DWACD" or "online WACD"'
    msg = ('scenario must be "Scenario-1", "Scenario-2", '
           '"Scenario-3" or "Scenario-4"')

    scenarios = ['Scenario-1', 'Scenario-2', 'Scenario-3', 'Scenario-4']
    assert scenario in scenarios, msg

    if scenario == 'Scenario-1':
        db = AsnycWASN(db_json).get_data_set_scenario_1()
    elif scenario == 'Scenario-2':
        db = AsnycWASN(db_json).get_data_set_scenario_2()
    elif scenario == 'Scenario-3':
        db = AsnycWASN(db_json).get_data_set_scenario_3()
    elif scenario == 'Scenario-4':
        db = AsnycWASN(db_json).get_data_set_scenario_4()

    if method == 'DWACD':
        sro_estimator = DynamicWACD()
    elif method == 'online WACD':
        sro_estimator = OnlineWACD()

    if method == 'DWACD':
        voice_activity_detector = VoiceActivityDetector(vad_threshold)

    num_examples = 3 * len(db)
    rmses = np.zeros(num_examples)
    for ex_id, example in enumerate(db):
        print(f'Process example {example["example_id"].split("_")[-1]}')
        ref_sig = load_audio(example['audio_path'][f'node_{ref_node_id}'])
        other_nodes = [i for i in range(4) if i != ref_node_id]
        for cnt, node_id in enumerate(other_nodes):
            sig = load_audio(example['audio_path'][f'node_{node_id}'])

            # Align the signals coarsely
            sig_sync, ref_sig_sync, offset = \
                coarse_sync(sig, ref_sig, len_sync=len_coarse_sync)

            # Estimate the sampling rate offset (SRO)
            if method == 'DWACD':
                activity_sig = voice_activity_detector(sig_sync)
                activity_ref_sig = voice_activity_detector(ref_sig_sync)
                sro_est = sro_estimator(
                    sig_sync, ref_sig_sync, activity_sig, activity_ref_sig
                )
            elif method == 'online WACD':
                settling_time = 40
                sro_est = sro_estimator(sig_sync, ref_sig_sync)
                sro_est[:settling_time-1] = sro_est[settling_time-1]

            # Get the ground truth SRO
            sro_sig = example['sro'][f'node_{node_id}']
            if isinstance(sro_sig, str):
                sro_sig = load_binary(sro_sig)
            sro_ref_sig = example['sro'][f'node_{ref_node_id}']
            if isinstance(sro_ref_sig, str):
                sro_ref_sig = load_binary(sro_ref_sig)
            block_offset = int(np.round(offset / 2048))
            if not np.isscalar(sro_sig):
                if block_offset > 0:
                    sro_sig = sro_sig[block_offset:block_offset+len(sro_est)]
                else:
                    sro_sig = sro_sig[:len(sro_est)]
            if not np.isscalar(sro_ref_sig):
                if block_offset < 0:
                    sro_ref_sig = \
                        sro_ref_sig[-block_offset:-block_offset + len(sro_est)]
                else:
                    sro_ref_sig = sro_ref_sig[:len(sro_est)]
            sro = sro_sig - sro_ref_sig

            # Calculate the estimation error
            max_len = np.minimum(len(sro), len(sro_est))
            rmses[3*ex_id+cnt] = \
                np.sqrt(np.mean((sro[:max_len] - sro_est[:max_len])**2))
            print(
                f'node {node_id}: RMSE = {np.round(rmses[3*ex_id+cnt], 2)} ppm'
            )
    print(f'\nAverage RMSE = {np.round(np.mean(rmses), 2)} ppm')
