from pathlib import Path

from paderbox.io import load_json, dump_json
from sacred import Experiment


ex = Experiment()


@ex.config
def config():
    database_path = None
    msg = \
        'You have to define the path where the downloaded database is stored.'
    assert database_path is not None, msg
    assert Path(database_path).is_dir(), database_path
    librispeech_path = None
    msg = \
        'You have to define the path where the LibriSpeech database is stored.'
    assert librispeech_path is not None, msg
    assert Path(librispeech_path).is_dir(), database_path
    json_path = None
    msg = \
        'You have to define the path where the database json should be stored.'
    assert json_path is not None, msg
    filename = Path(json_path)
    msg = f'Json file must end with ".json" and not "{filename.suffix}"'
    assert filename.suffix == '.json', msg


@ex.capture
def complete_source_information(source_description,
                                example_id,
                                setups,
                                rir_root,
                                librispeech_path):
    source_description["audio_path"] = \
        Path(librispeech_path).joinpath(source_description["audio_path"])
    source_position = source_description['source_position']
    rir_path = rir_root.joinpath(f'{example_id}/src_{source_position}/')
    source_description['rirs'] = \
        {f'node_{i}': rir_path.joinpath(f'node_{i}.wav') for i in range(4)}
    source_description['source_position'] = \
        setups[example_id]['source_position'][source_position]
    return source_description


@ex.automain
def create_json(database_path, json_path):
    database_path = Path(database_path)
    rir_root = database_path.joinpath('rirs/')
    setups = load_json(Path(database_path).joinpath('setups.json'))
    simulation_descriptions = \
        load_json(Path(database_path).joinpath('simulation_descriptions.json'))
    for scenario in simulation_descriptions.values():
        for example_id, example in scenario.items():
            for node_id, sro in example['sro'].items():
                if isinstance(sro, str):
                    example['sro'][node_id] = database_path.joinpath(sro)
            example['node__position'] = setups[example_id]['node_position']
            example['node_orientation'] = \
                setups[example_id]['node_orientation']
            example['environment'] = setups[example_id]['environment']
            example['src_diary'] = [
                complete_source_information(
                    source, example_id, setups, rir_root
                )
                for source in example['src_diary']
            ]
    db = {'datasets': simulation_descriptions}
    dump_json(db, json_path, sort_keys=False)
