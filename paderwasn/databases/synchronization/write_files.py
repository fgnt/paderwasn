from pathlib import Path

from paderbox.io import dump_audio, dump_json, load_json
from sacred import Experiment
from tqdm.auto import tqdm
from paderwasn.databases.synchronization.database import AsyncWASN
from paderwasn.databases.synchronization.audio_generation import generate_audio

ex = Experiment()


@ex.config
def config():
    json_path = None
    msg = 'You have to define the path where the database json is stored.'
    assert json_path is not None, msg
    data_root = None
    msg = 'You have to define the path where the files should be stored.'
    assert data_root is not None, msg
    json_file_db_path = None
    msg = ('You have to define the path where the'
           'database json (with written files)is stored.')
    assert json_file_db_path is not None, msg


@ex.automain
def generate_file_db(json_path, json_file_db_path, data_root):
    def generate_audio_all_nodes(example):
        for node_id in range(4):
            example = generate_audio(
                example, f'node_{node_id}', std_sensor_noise=9.774e-5,
                sig_len=4800000, single_channel=True
            )
        return example

    db = load_json(json_path)
    data_root = Path(data_root)
    async_wasn = AsyncWASN(json_path)
    for scenario in range(1, 5):
        examples = async_wasn.get_dataset(f'scenario_{scenario}')
        examples = examples.map(generate_audio_all_nodes)
        pbar = tqdm(total=len(examples))
        pbar.set_description(f'Create Scenario-{scenario}')
        for example in examples:
            ex_id = example["example_id"]
            audio_paths = {}
            for node_id in range(4):
                filename = data_root.joinpath(
                    f'scenario_{scenario}/{ex_id}/node_{node_id}.wav'
                )
                audio_paths[f'node_{node_id}'] = filename
                filename.parent.mkdir(parents=True, exist_ok=True)
                dump_audio(example['audio_data'][f'node_{node_id}'],
                           filename,
                           normalize=False)
            db['datasets'][f'scenario_{scenario}'][ex_id]['audio_path'] = \
                audio_paths
            src_diary = \
                db['datasets'][f'scenario_{scenario}'][ex_id]['src_diary']
            for source in src_diary:
                source.pop('audio_path')
                source.pop('rirs')
            pbar.update(1)
    dump_json(db, json_file_db_path, sort_keys=False)
