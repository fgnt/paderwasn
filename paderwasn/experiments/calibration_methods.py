"""
Example call:
python -m paderwasn.experiments.calibration_methods with 'iter_dsm_simple'
"""
import numpy as np
from sacred import Experiment
from time import time
from paderbox.math.directional import wrap
from paderwasn.databases.geometry_calibration.database import CalibrationDataSet
from paderwasn.databases.geometry_calibration.utils import get_node_positions, \
    get_node_orientations, get_doas, get_sn_dists
from paderwasn.geometry_calibration.garde import est_geometry as calib_garde
from paderwasn.geometry_calibration.iterative_dsm import \
    est_geometry as calib_iter_dsm
from paderwasn.geometry_calibration.utils import est_rot_mat, \
    map2ref, rot_mat2rot_angle


ex = Experiment('Evaluate calibration method')


@ex.config
def config():
    t60 = 500
    n_obs = 100
    n_reps = 100
    method = None
    calib_conf = None


@ex.named_config
def iter_dsm():
    method = 'iter_dsm'
    calib_conf = {
        'outlier_percent': 0.5,
        'conv_th': 1e-3,
        'max_iter': 100,
        'wls': True,
        'wls_src_loc': True
    }


@ex.named_config
def iter_dsm_simple():
    method = 'iter_dsm_simple'
    calib_conf = {
        'outlier_percent': 0,
        'conv_th': 1e-3,
        'max_iter': 100,
        'wls': False,
        'wls_src_loc': False
    }


@ex.named_config
def garde():
    method = 'garde'
    calib_conf = {
        'n_iter': 10,
        'n_generations': 30,
        'generation_spread': 2,
        'outlier_percent': 0.6,
        'lr': 0.4
    }


def calc_orient_error(node_pos, node_orients, gt_node_pos, gt_node_orients):
    rot_mat = est_rot_mat(node_pos, gt_node_pos)
    ref_orient = rot_mat2rot_angle(rot_mat)
    orient_errors = \
        np.abs(wrap(wrap(node_orients + ref_orient) - gt_node_orients))
    return orient_errors / np.pi * 180


@ex.automain
def eval_method(method, calib_conf, t60, n_obs, n_reps):
    msg = 'You have to choose the geometry calibration method.'
    assert method is not None, msg
    assert method in ['iter_dsm', 'iter_dsm_simple', 'garde']

    scenarios = CalibrationDataSet().get_dataset(f't60_{t60}')
    n_nodes = get_node_orientations(scenarios[0]).size
    comp_times = np.zeros((n_reps, len(scenarios)))
    pos_errors = np.zeros((n_reps, len(scenarios), n_nodes))
    if method != 'garde':
        orient_errors = np.zeros((n_reps, len(scenarios), n_nodes))
    for scenario_id, scenario in enumerate(scenarios):
        for rep in range(n_reps):
            print(f'scenario {scenario_id + 1}/{len(scenarios)}; '
                  f'repetition {rep + 1}/{n_reps}')
            gt_node_pos = get_node_positions(scenario)[:2]
            gt_node_orients = get_node_orientations(scenario)
            doas = get_doas(scenario)[:n_obs]
            dists = get_sn_dists(scenario)[:n_obs]
            if method in ['iter_dsm', 'iter_dsm_simple']:
                start_time = time()
                node_pos, node_orients, _, _ = \
                    calib_iter_dsm(doas, dists, **calib_conf)
                end_time = time()
                orient_errors[rep, scenario_id] = calc_orient_error(
                    node_pos, node_orients, gt_node_pos, gt_node_orients
                )
                node_pos_fixed = map2ref(node_pos, gt_node_pos)
            else:
                start_time = time()
                node_pos, _ = calib_garde(dists, **calib_conf)
                end_time = time()
                node_pos_fixed = \
                    map2ref(node_pos, gt_node_pos, allow_reflection=True)
            pos_errors[rep, scenario_id] = \
                np.linalg.norm(node_pos_fixed - gt_node_pos, axis=0)
            comp_times[rep, scenario_id] = end_time - start_time
    print('\n')
    print('Results:')
    print(f'MAE node positions: {np.mean(pos_errors)}m')
    if method in ['iter_dsm', 'iter_dsm_simple']:
        print(f'MAE node orientations: {np.mean(orient_errors)}°')
    print(f'Average computing time: {np.mean(comp_times * 1000)}ms')
