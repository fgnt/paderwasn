import numpy as np


def get_node_positions(scenario):
    assert 'geometry' in scenario.keys()
    assert 'node_positions' in scenario['geometry'].keys()
    return np.asarray(scenario['geometry']['node_positions'])


def get_node_orientations(scenario):
    assert 'geometry' in scenario.keys()
    assert 'node_orientations' in scenario['geometry'].keys()
    return np.asarray(scenario['geometry']['node_orientations'])


def get_source_positions(scenario):
    assert 'acoustic_sources' in scenario.keys()
    sources = scenario['acoustic_sources']
    src_positions = np.zeros((3, len(sources)))
    for src_id, src in enumerate(sources):
        src_positions[:, src_id] = src['source_position']
    return src_positions


def get_doas(scenario, key='estimates'):
    assert 'observations' in scenario.keys()
    assert key in scenario['observations'].keys()
    assert ('doas' in scenario['observations'][key].keys())
    return np.asarray(scenario['observations'][key]['doas'])


def get_sn_dists(scenario, key='estimates'):
    assert 'observations' in scenario.keys()
    assert key in scenario['observations'].keys()
    assert ('sn_distances' in scenario['observations'][key].keys())
    return np.asarray(scenario['observations'][key]['sn_distances'])
