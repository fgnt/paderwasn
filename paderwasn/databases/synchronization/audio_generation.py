import warnings

import numpy as np
from paderbox.io import load_audio

from paderwasn.databases.synchronization.utils import \
    load_binary, reverb_signal
from paderwasn.synchronization.simulation import sim_sro


def generate_audio(example,
                   node_id,
                   std_sensor_noise,
                   sig_len=None,
                   single_channel=False,
                   max_sro=400):
    """
    Generates the audio signal recorded by a sensor node using the given
    simulation description. This function is typically used as map function in
    combination with the lazy_data set package.

    Args:
        example:
            Example dictionary specifying how to generate the audio signal
        node_id:
            Integer identifying the sensor node for which the recorded signal
            should be simulated.
        std_sensor_noise:
            Standard deviation of the simulated sensor noise.
        sig_len:
            Length (in samples) of the signal to be created.
        single_channel:
            Boolean specifying if all microphone signals should be simulated.
            If true only one channel is simulated. Otherwise, all microphone
            channels are simulated.
        max_sro:
            Expected maximum value for the sampling rate offset (SRO)
    Returns:
        Example dictionary with additionally added audio signal
    """
    min_sto = np.minimum(np.min([sto for sto in example['sto'].values()]), 0)
    stos = {node_id: sto - min_sto for node_id, sto in example['sto'].items()}
    src_diary = example['src_diary']

    if single_channel:
        num_channels = 1
    else:
        num_channels = len(load_audio(src_diary[0]['rirs']['node_0']))

    if sig_len is not None:
        min_sig_len = sig_len
        max_sro_delay = int(np.ceil(max_sro * 1e-6 * sig_len))
        min_sig_len += \
            max_sro_delay + np.max([np.abs(sto) for sto in stos.values()])
        if min_sig_len > example['src_diary'][-1]['offset']:
            min_sig_len = example['src_diary'][-1]['offset']
            warnings.warn(
                'Specified signal length is larger than maximum signal length'
                'defined by the source diary. The signal length is set to'
                'maximum signal length defined by the source diary'
            )
    else:
        min_sig_len = src_diary[-1]['offset']

    audio_data = np.zeros((num_channels, min_sig_len))

    for source in src_diary:
        onset = source['onset']
        clean_audio = load_audio(source['audio_path'])
        rirs = load_audio(source['rirs'][node_id])
        if single_channel:
            rirs = rirs[0, None]
        reverberant_audio = reverb_signal(clean_audio, rirs)
        if onset + reverberant_audio.shape[-1] > audio_data.shape[-1]:
            missing_len = \
                onset + reverberant_audio.shape[-1] - audio_data.shape[-1]
            audio_data = \
                np.pad(audio_data, ((0, 0), (0, missing_len)), mode='constant')
            audio_data[:, onset:onset + reverberant_audio.shape[-1]] += \
                reverberant_audio
            break
        audio_data[:, onset:onset + reverberant_audio.shape[-1]] += \
            reverberant_audio

    audio_data = audio_data[:, stos[node_id]:]
    sro = example['sro'][node_id]
    if isinstance(sro, str):
        sro = load_binary(sro)
    audio_data = np.asarray([sim_sro(ch, sro) for ch in audio_data])
    audio_data += np.random.normal(0, std_sensor_noise, size=audio_data.shape)

    if sig_len is not None:
        audio_data = audio_data[:, :sig_len]

    if 'audio_data' in example.keys():
        example['audio_data'][node_id] = audio_data
    else:
        example['audio_data'] = {node_id: audio_data}
    return example
