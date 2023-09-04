from einops import rearrange

import numpy as np
from scipy import signal
import paderbox as pb
from pb_bss.extraction.beamformer import get_power_spectral_density_matrix
from pb_bss.extraction.beamformer import get_mvdr_vector_souden

from paderwasn.synchronization.time_shift_estimation import max_time_lag_search
from paderwasn.source_extraction.eigen_decomposition import get_dominant_eigenvector
from paderwasn.source_extraction.eigen_decomposition import get_eigenvalue
from paderwasn.source_extraction.segmentation import get_activities_time_domain
from paderwasn.source_extraction.segmentation import bridge_pauses_transcription
from paderwasn.source_extraction.segmentation import segment_by_activity


def _update_scms(
        scm_buffer, scms_beam,stft_buffer, masks, activities, block_id,
        block_size, block_shift_scm, shift_ratio_beam, alpha
):
    """
    Update the spatial covariance matrices (SCMs) of all sources (speakers
    + noise) based on the given time-frequency masks and activity estimates

    Returns:
        scm_buffer:
            History of block-wise estimates SCMs of all sources
        scms_beam:
            SCMs used for beamforming  (updated in a block-online manner)
    """
    block_begin = block_id * block_shift_scm
    block_end = block_begin + block_size
    scm_buffer = np.roll(scm_buffer, -1, axis=0)
    scm_buffer[-1] = 0
    for src_id in range(len(masks)):
        if np.isclose(np.sum(scms_beam[src_id]), 0):
            # If there is no estimate for the SCMs enable SCM estimation for a
            # block with less activity. This is a tradeoff between getting an
            # estimate for the SCMs as soon as possible and having a more
            # robust estimate.
            act_th = block_size // 4
        else:
            # If there is an estimate for the SCMs the SCMs will be updated
            # only if the source is active in most of the time within a block
            # to get robust estimates.
            act_th = int(block_size * 3 / 4)
        if np.sum(activities[src_id, block_begin:block_end]) >= act_th:
            # Calculate the dyade-product between the masked vectors of
            # stacked STFTs of all channels.
            stft_block = \
                stft_buffer[..., activities[src_id, block_begin:block_end]]
            src_mask = masks[src_id, None, ..., block_begin:block_end]
            src_mask = \
                src_mask[..., activities[src_id, block_begin:block_end]]
            scm_update = get_power_spectral_density_matrix(
                rearrange(stft_block, 'c f t -> f c t')
                * rearrange(src_mask, 'c f t -> f c t')
            )
            # Normalize the SCMs w.r.t. the amount of frames with source
            # activity
            scm_update *= block_size
            scm_update /= np.sum(activities[src_id, block_begin:block_end])
            scm_buffer[-1, src_id] = scm_update.copy()

            # Update the SCMs used to calculate the beamformer coefficients
            # in a block-online manner
            if block_id % shift_ratio_beam == 0:
                scms_beam[src_id] = \
                    alpha * scms_beam[src_id] + (1 - alpha) * scm_update
    return scm_buffer, scms_beam


def _apply_beamformer(
        bf_output, target_speaker, block_id, stft_buffer, scms_beam,
        activities, block_size, shift_ratio_beam, block_shift_scm
):
    """
    Estimate the coefficients of a time-varying minimum-variance
    distortionless response (MVDR) beamformer in the formulation of
    [Souden2010MVDR] and apply it to the current block of the input signal
    to extract the signal of the given target speaker.

    @article{Souden2010MVDR,
      title={On optimal frequency-domain multichannel linear filtering for
             noise reduction},
      author={Souden, Mehrez and Benesty, Jacob and Affes, Sofi{\`e}ne},
      journal={IEEE Transactions on audio, speech, and language processing},
      volume={18},
      number={2},
      pages={260--276},
      year={2010},
      publisher={IEEE}
    }
    """
    if block_id != 0:
        block_begin = ((block_id - shift_ratio_beam) * block_shift_scm
                       + block_size)
        block_end = block_id * block_shift_scm + block_size
    else:
        block_begin = 0
        block_end = block_size
    if np.sum(activities[target_speaker, block_begin:block_end]) > 0:
        target_scm = scms_beam[target_speaker].copy()

        # Interference-SCM as sum of the SCMs of all interfering sources, which
        # are active in the current block.
        select = np.sum(activities[:, block_begin:block_end], -1) > 0
        select[target_speaker] = False
        interference_scm = np.sum(scms_beam[select], 0)
        interference_scm += \
            np.finfo(np.float64).eps * np.eye(stft_buffer.shape[0])[None]

        bf_vec = get_mvdr_vector_souden(
            target_scm, interference_scm,
            ref_channel=0, eps=np.finfo(np.float64).tiny
        )
        if block_id != 0:
            new_block = stft_buffer[..., -(block_end - block_begin):]
        else:
            new_block = stft_buffer
        bf_output[block_begin:block_end] = np.einsum(
            'f c, c f t -> t f', np.conj(bf_vec), new_block
        )
    return bf_output


def _get_coherence_product_update(
        scm_buffer, masks, block_id, num_chs, num_srcs, fft_size, block_size,
        block_shift_scm, size_shift_ratio, noise_class
):
    """
    Get updates for the average complex-conjugated products
    of consecutive coherence functions
    """
    updated = False
    coherence_prod_update = \
        np.zeros((fft_size // 2 + 1, num_chs, num_chs), np.complex128)
    for src_id in range(num_srcs):
        # If a source is not active in a block, its SCM is set to zero. Thus,
        # the activity of a source within a block is indicated by a SCM with
        # non-zero entries.
        activity_block = np.isclose(np.sum(abs(scm_buffer[0, src_id])), 0)
        activity_old_block = np.isclose(np.sum(abs(scm_buffer[-1, src_id])), 0)

        # A speaker needs to be active in the current block and the previous
        # block involved in the calculation of the complex-conjugated product
        # of consecutive coherence functions so that he is considered when
        # calculating the update of the average porduct of coherence functions.
        if (not activity_block) and (not activity_old_block):
            # The SCM belonging to the noise-class is not used to estimate
            # the sampling rate offsets.
            if src_id != noise_class:
                updated = True

                # Derive pair-wise coherences of previous block from the SCMs.
                src_scms_old = scm_buffer[0, src_id].copy()
                # The auto power spectral densities are on the main diagonal
                # of the SCMs.
                auto_psds = np.diagonal(
                    np.maximum(src_scms_old, np.finfo(np.float64).eps),
                    axis1=-2, axis2=-1
                )
                denominator = np.einsum(
                    '... c, ... d -> ... c d', auto_psds, auto_psds
                )
                denominator = np.sqrt(denominator)
                coherences_old = src_scms_old / denominator

                # Derive pair-wise coherences of current block from the SCMs.
                src_scms = scm_buffer[-1, src_id].copy()
                auto_psds = np.diagonal(
                    np.maximum(src_scms, np.finfo(np.float64).eps),
                    axis1=-2, axis2=-1
                )
                denominator = np.einsum(
                    '... c, ... d -> ... c d', auto_psds, auto_psds
                )
                denominator = np.sqrt(denominator)
                coherences = src_scms / denominator

                # Due to the masking in combination with the normalization by
                # the auto-power spectral densities when calculating the
                # coherence some frequencies might get a high weight although
                # the source is only active within a few frames within a block
                # for a frequency. This might lead to inaccurate phase
                # estimates and, consequently, inaccurate SRO estimates,
                # Therefore, the complex-conjugated product of consecutive
                # coherence functions is weighted based on the amount of frames
                # per frequency in which a source is considered to be dominant.
                block_begin = block_id * block_shift_scm
                block_end = block_begin + block_size
                # A source is considered to be dominant within a time-frequency
                # bin if the corresponding value of the time-frequency mask
                # (the masks for all sources, i.e., all speakers plus noise,
                # have to sum up to one for each time-frequency bin) is larger
                # than 0.9. The frequency-weight of the coherence of the
                # current block is given by the average amount of frames in
                # which the source is dominant during the current block.
                freq_weights = np.mean(
                    masks[src_id, ..., block_begin:block_end] > .9, -1
                )
                block_begin = \
                    (block_id - size_shift_ratio) * block_shift_scm
                block_end = block_begin + block_size
                freq_weights_old = np.mean(
                    masks[src_id, ..., block_begin:block_end] > .9, -1
                )
                # The overall frequency-weight for the complex-conjugated
                # product of consecutive coherence functions is given by the
                # geometric mean of the frequency-weight of the current block
                # and the previous block involved in the calculation of the
                # product of coherence functions.
                freq_weights = np.sqrt(freq_weights * freq_weights_old)

                # The overall update of the average complex-conjugated product
                # of consecutive coherence functions is given by the (weighted)
                # sum of the  products of consecutive coherence functions of
                # all speakers being active.
                coherence_products = coherences * coherences_old.conj()
                coherence_prod_update += \
                    coherence_products * freq_weights[:, None, None]
    return coherence_prod_update, updated


def _estimate_sros(
        avg_coherence_prod, mic_groups, multi_ch_sro_est, eigen_vector,
        block_size, frame_shift, num_chs
):
    """
    Derive the sampling rate offset (SRO) estimates from the average
    complex-conjugated product of consecutive coherence functions
    """
    sros = np.zeros(num_chs)

    # Since the SROs can only be estimated w.r.t. a reference channel in
    # practice, the SRO is here estimated  w.r.t. the first channel of the
    # first microphone group.
    ref_mic_group = mic_groups[0]
    if len(ref_mic_group) > 1:
        ref_mic = ref_mic_group[0]
    else:
        ref_mic = ref_mic_group

    # Estimate the SRO either pair-wisely w.r.t. a reference channel or in a
    # multi-channel fashion considering the relationships between all
    # pair-wise SROs
    if multi_ch_sro_est:
        # Estimate the SRO from the dominant eigenvector of the matrix of all
        # pair-wise products of consecutive coherence functions
        eigen_vector = get_dominant_eigenvector(
            avg_coherence_prod, eigen_vector
        )
        # Scale the dominant eigenvector using the corresponding eigenvalue to
        # retain the (signal-to-noise ratio related) frequency-weighting of the
        # average complex-conjugated weighting.
        eigen_val = get_eigenvalue(avg_coherence_prod, eigen_vector)
        scale = eigen_val
        scale /= np.maximum(
            np.linalg.norm(eigen_vector, axis=-1),
            np.finfo(np.float64).eps
        )
        eigen_vector *= scale[..., None]
        # Force that the phase of the entry belonging to the reference channel
        # is zero.
        eigen_vector *= eigen_vector[:, ref_mic, None].conj()

        for i, mic_ids in enumerate(mic_groups[1:]):
            mic_id = mic_ids[0]
            # Estimate the SRO-induced time shift between two channels based
            # on the corresponding entry of the dominant eigenvector. Based on
            # this value the SRO estimate is derived by dividing the time shift
            # by the temporal distance between the coherence functions involved
            # in the calculation of the product of consecutive coherence
            # functions.
            sro_est = (max_time_lag_search(eigen_vector[:, mic_id])
                       / (block_size * frame_shift))
            for mic_id in mic_ids:
                sros[mic_id] = sro_est
        return sros, eigen_vector
    else:
        for i, mic_ids in enumerate(mic_groups[1:]):
            mic_id = mic_ids[0]
            avg_coh_prod = avg_coherence_prod[:, ref_mic, mic_id]
            # Estimate the SRO-induced time shift between a channel and the
            # reference channel based on the corresponding pair-wise average
            # complex-conjugated product of consecutive coherence functions.
            # Based on this value the SRO estimate is derived by dividing the
            # time shift by the temporal distance between the coherence
            # functions involved in the calculation of the product of
            # consecutive coherence functions.
            sro_est = (-max_time_lag_search(avg_coh_prod)
                       / (block_size * frame_shift))
            for mic_id in mic_ids:
                sros[mic_id] = sro_est
        return sros


def synchronizing_block_online_mvdr(
        target_speaker, sigs, masks, activities, mic_groups=None,
        activities_segmentation=None, multi_ch_sro_est=True, noise_class=-1,
        block_shift_bf=32, block_shift_scm=8, block_size=32, fft_size=1024,
        frame_size=1024, frame_shift=256, alpha=.95, settling_time=20
):
    """
    Joint sampling rate offset synchronization and source extraction via
    beamforming as proposed in [Gburrek23]

    @inproceedings{Gburrek23,
       author={Gburrek, Tobias and Schmalenstroeer, Joerg
               and Haeb-Umbach, Reinhold},
       booktitle = {31st European Signal Processing Conference (EUSIPCO)},
       pages = {1--5},
       title = {{On the Integration of Sampling Rate Synchronization and
                 Acoustic Beamforming}},
       year = {2023},
    }

    Args:
        target_speaker (in):
            Identifier of the speaker whose signal should be extracted.
        sigs (np.ndarray):
            Speech mixture (Shape: (number of channels x signal length))
        masks (np.ndarray):
            Time-frequency masks (Shape:
            (number of speakers + 1 x FFT-size / 2 + 1 x number of frames))
        activities (np.ndarray):
            Activity of the sources (speakers + noise) used for SCM estimation
            and calcuation of the time-varying beamformer coefficients (Shape:
            (number of speakers + 1 x number of frames))
        mic_groups (list of lists):
            Groups of microphones sharing the same SRO. The first group defines
             the refrence for SRO estimation.
        activities_segmentation (np.ndarray):
            Activity of the target speaker used to segment the enhanced signal
            into utterances by discarding silence (Shape: (number of frames)).
            If None (default) the activity of the target speaker is taken
            from ``activities´´.
        multi_ch_sro_est (boolean):
            Flag indicating whether the multi-channel version of the SRO
            estimator should be used.
        noise_class (int):
            Identifies which mask/activity correspond to the noise. If -1
            (default) the last mask/activity is assumed to belong to the noise.
        block_shift_bf (int):
            Amount of frames between two updates of the SCM used for
            beamforming and the beamformer coefficients in a block-online
            manner. Need to be larger than or equal to block_shift_scm.
        block_shift_scm (int):
            Amount of frames between two updates of the SCM used for SRO
            estimation. This also defines the rate of the SRO estimates.
        block_size (int):
            Amount of frames involved in a block-wise update of the frames and
            beamformer coffeicients,
        fft_size (int):
            FFT size used to calculate the STFT.
        frame_size (int):
            Frame size used to calculate the STFT.
        frame_shift (int):
            Frame shift used to calculate the STFT.
        alpha (float):
            Smoothing factor used for the autoregressive smoothing for time
            averaging when estimating SCMs.
        settling_time (int):
            Amount of block with suitable source activity after which the SRO
            is estimated for the first time and the compensation for SROs is
            started.
    Returns:
        enhanced_utts (list of np.ndarrays):
            List of extracted ``utterances´´ of the given target speaker.
            An ``utterance´´ is defined by the section of contiguous activity
            of the target speaker.
        sro_trajectories (list of np.ndarrays):
            List of estimated SRO trajectories per group of microphones sharing
             the same SRO.
    """
    msg = 'You have to specify which microphones that share the same SRO.'
    assert mic_groups is not None, msg
    msg = 'mic_groups has to be a list of lists.'
    for group in mic_groups:
        assert isinstance(group,list), msg
    msg = (f'block_size ({block_size}) must be an integer multiple of '
           f'block_shift_scm ({block_shift_scm})')
    assert block_size % block_shift_scm == 0, msg
    msg = (f'block_shift_bf ({block_shift_bf}) has to be larger than '
           f'or equal to block_shift_scm ({block_shift_scm}).')
    assert block_shift_bf >= block_shift_scm, msg

    k = np.arange(fft_size // 2 + 1)
    window = signal.windows.blackman(frame_size + 1)[:-1]
    if noise_class == -1:
        noise_class = len(masks) - 1

    num_srcs = len(masks)
    num_chs = len(sigs)
    num_frames = (sigs.shape[-1] - frame_size) // frame_shift + 1
    num_blocks = \
        int(np.ceil((num_frames - block_size) // block_shift_scm + 1))

    # Ratio between the block shift used for beamforming and SCM estimation.
    # This can be used to update the beamformer less often than the SCMs.
    # Updating a beamformer too often might be detrimental for a subsequent
    # automatic speech recognition system.
    shift_ratio_beam = block_shift_bf // block_shift_scm

    size_shift_ratio = block_size // block_shift_scm
    scm_buffer = np.zeros(
        (size_shift_ratio + 1, num_srcs, fft_size // 2 + 1, num_chs, num_chs),
        np.complex128
    )
    scms_beam = np.zeros(
        (num_srcs, fft_size // 2 + 1, num_chs, num_chs), np.complex128
    )

    stft_buffer = \
        np.zeros((len(sigs), fft_size // 2 + 1, block_size), np.complex128)
    bf_output = np.zeros(
        (num_frames, fft_size // 2 + 1), np.complex128
    )

    sro_trajectories = [[] for _ in range(len(mic_groups) - 1)]
    sros = np.zeros(num_chs)
    sro_buffer = np.zeros((len(mic_groups) - 1, len(sigs)))
    sro_induced_delay = np.zeros((num_chs, block_size))
    integer_delay = np.zeros((num_chs, block_size))
    avg_coherence_prod = np.zeros((fft_size, num_chs, num_chs), np.complex128)
    eigen_vector = None
    is_resampled = np.zeros(size_shift_ratio, bool)

    update_cnt = 0
    started = False
    reached_end = False
    # Update the buffer of the STFT using a analysis window with non-uniform
    # shift to compensate for the integer-part of the SRO-induced shift.
    for block_id in range(num_blocks):
        block_begin = block_id * block_shift_scm
        for l in range(block_size):
            for c in range(num_chs):
                # Check whether the end of the signal is reached.
                start_sample = block_begin * frame_shift
                end_sample = start_sample + l * frame_shift + frame_size
                end_sample += int(integer_delay[c, l])
                if end_sample > sigs.shape[-1]:
                    reached_end = True
                    break

                # Get the start of the current block. Note that the integer
                # time shift induced by the estimated SRO is compensated for by
                # a non-uniform shift of the analysis window of the STFT.
                start = (block_begin * frame_shift + l * frame_shift
                         + int(integer_delay[c, l]))
                stft_buffer[c, :, l] = np.fft.rfft(
                    sigs[c, start:start+frame_size] * window
                )
        if reached_end:
            break

        # After compensating for the integer time shift induced by the SRO a
        # sub-sample delay remains, which is compensated for by a phase shift
        # in the frequency domain.
        remaining_delay = integer_delay - sro_induced_delay
        delay_vect = np.exp(-1j * 2 * np.pi * k[None, None]
                            * remaining_delay[:, :, None] / fft_size)
        stft_buffer *= rearrange(delay_vect, 'c t f -> c f t')

        scm_buffer, scms_beam = _update_scms(
            scm_buffer, scms_beam,stft_buffer, masks, activities, block_id,
            block_size, block_shift_scm, shift_ratio_beam, alpha
        )

        if block_id % shift_ratio_beam == 0:
            bf_output = _apply_beamformer(
                bf_output, target_speaker, block_id, stft_buffer, scms_beam,
                activities, block_size, shift_ratio_beam, block_shift_scm
            )

        if block_id < size_shift_ratio:
            continue

        coherence_prod_update, updated = _get_coherence_product_update(
            scm_buffer, masks, block_id, num_chs, num_srcs, fft_size,
            block_size, block_shift_scm, size_shift_ratio, noise_class
        )
        if updated:
            # Apply a phase shift to the update of the coherence_product update
            # to undo the resampling based on the current SRO estimates. This
            # enables to average the coherence product over time without need
            # for a controller. Note that the estimated SRO consequently is not
            # driven to zero although the signal and the SCMs are resampled.
            undo_delay_vect = np.exp(1j * 2 * np.pi * k[None] * sros[:, None]
                                     * block_size * frame_shift / fft_size)
            coherence_prod_update *= np.einsum(
                'c f, d f -> f c d', undo_delay_vect.conj(), undo_delay_vect
            )

            coherence_prod_update = np.concatenate(
                [coherence_prod_update[:-1],
                 np.conj(coherence_prod_update)[::-1][:-1]],
                0
            )
            if not started or (is_resampled[0] and is_resampled[-1]):
                avg_coherence_prod = (
                    alpha * avg_coherence_prod
                    + (1 - alpha) * coherence_prod_update
                )
            update_cnt += 1

        sro_buffer = np.roll(sro_buffer, -1, axis=0)
        if update_cnt >= settling_time:
            if multi_ch_sro_est:
                sros, eigen_vector = _estimate_sros(
                    avg_coherence_prod, mic_groups, multi_ch_sro_est,
                    eigen_vector, block_size, frame_shift, num_chs
                )
            else:
                sros = _estimate_sros(
                    avg_coherence_prod, mic_groups, multi_ch_sro_est,
                    eigen_vector, block_size, frame_shift, num_chs
                )

        sro_buffer[-1] = sros.copy()
        if update_cnt == settling_time and not started:
            avg_coherence_prod = \
                np.zeros((fft_size, num_chs, num_chs), np.complex128)
            update_cnt = 0
            started = True
        if started:
            is_resampled = np.roll(is_resampled, -1)
            is_resampled[-1] = True
        for i, mic_ids in enumerate(mic_groups[1:]):
            mic_id = mic_ids[0]
            sro_trajectories[i].append(sros[mic_id] * 1e6)
        # Update the SRO-induced time shift.
        for _ in range(block_shift_scm):
            sro_induced_delay = np.roll(sro_induced_delay, -1, axis=1)
            sro_induced_delay[:, -1] = \
                sro_induced_delay[:, -2] + sros * frame_shift
        integer_delay = np.round(sro_induced_delay)
    enh_sig = pb.transform.istft(
        bf_output, size=fft_size, shift=frame_shift, window_length=frame_size
    )

    # Segment the signal into single ``utterances´´ by discarding periods in
    # time without activity of the target speaker.
    if activities_segmentation is None:
        activities_segmentation = activities[target_speaker]
    target_activity = get_activities_time_domain(
        activities_segmentation[None], frame_size=frame_size,
        frame_shift=frame_shift
    )
    target_activity = target_activity.squeeze(0)
    target_activity = bridge_pauses_transcription(target_activity)
    enhanced_utts, *_ = segment_by_activity(enh_sig, target_activity)
    return enhanced_utts, sro_trajectories
