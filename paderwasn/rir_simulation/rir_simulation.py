"""
RIR simulation code as described in
Gburrek, T., Meise, A., Schmalenstroeer, J., Haeb-Umbach, R.:
“Diminishing Domain Mismatch for DNN-Based Acoustic Distance Estimation
via Stochastic Room Reverberation Models”,
accepted to IWAENC 2024
"""

from enum import Enum
import warnings

import numpy as np
import pyroomacoustics as pra
from pyroomacoustics.directivities import DirectionVector, CardioidFamily
from scipy.signal.windows import hann


class DirectivityPattern(Enum):
    """
    Common cardioid patterns and corresponding coefficient for the expression:

        r = a (1 + cos(theta)),

    where `a` is the coefficient that determines the cardioid pattern and
    `r` yields the gain at angle `theta` (see
    pyroomacoustics.directivities.DirectivityPattern).

    """
    FIGURE_EIGHT = 0
    HYPERCARDIOID = 0.25
    CARDIOID = 0.5
    SUBCARDIOID = 0.75
    OMNI = 1.0
    SUPERCARDIOID = 1 / (1 + 1 / 0.5736)


pattern_dispatcher = {
    'cardioid': DirectivityPattern.CARDIOID,
    'hypercardioid': DirectivityPattern.HYPERCARDIOID,
    'supercardioid': DirectivityPattern.SUPERCARDIOID,
    'subcardioid': DirectivityPattern.SUBCARDIOID,
    'omnidirectional': DirectivityPattern.OMNI,
}


def high_pass(rir):
    """
        Original high-pass filter as proposed by Allen and Berkley
        (See RIR-Generator of Habets)

        Args:
             rir (array_lke): input room impulse response to be filtered

        Returns: filtered room impulse response
    """
    W = 2 * np.pi * 100 / 16000
    R1 = np.exp(-W)
    B1 = 2 * R1 * np.cos(W)
    B2 = -R1 * R1
    A1 = -(1 + R1)

    Y = np.zeros(3)
    for idx in range(len(rir)):
        X0 = rir[idx]
        Y[2] = Y[1]
        Y[1] = Y[0]
        Y[0] = B1 * Y[1] + B2 * Y[2] + X0
        rir[idx] = Y[0] + A1 * Y[1] + R1 * Y[2]
    return rir


def calc_target_drr(room_dimensions, alpha, beta, t60, directivity_attenuation,
                    distance):
    """
    Calculate the target direct-to-reverberant energy ratio (DRR) for the given
    room, distance between the source and microphone and directivity parameters
    according to the proposed model.

    Args:
        room_dimensions (array-like):
            Size of the room with shape (dimension)

        alpha (float):
            Directivity factor of acoustic source

        beta (float):
            Directivity factor of microphone

        t60 (float):
            T60 time of the room in seconds

        directivity_attenuation (float):
            Source’s directional response based on source directivity,
            azimuth and elevation of the source

        distance (float):
            Distance between microphone position and acoustic source

    Returns:
        Target DRR of the RIR to be generated
    """
    room_volume = np.prod(room_dimensions)
    critical_distance = \
        0.1 * np.sqrt(alpha * beta) * np.sqrt(room_volume / (np.pi * t60))
    target_drr = (
        directivity_attenuation ** 2 * critical_distance ** 2
        / distance ** 2
    )
    return target_drr


def calc_window(delta, kappa, toa_direct, rir_len, sample_rate):
    """
    Calculate weighting function for stochastic part of the RIR to achieve a
    smooth fade-in of the late reflections.

    Args:
        delta (float):
            Time constant of exponential envelope needed to determine the
            length of the transition from 0 to 1.

        kappa (float):
           Scales the length of the fade-in of the stochastic part of the RIR.
           Lower values cause smoother transition.

        toa_direct(int):
            Index of direct path for current RIR example.

        rir_len (int):
            Length of RIR to be generated in samples

        sample_rate (int):
            Sample rate

    Returns:
        Padded weighting function of the shape (rir_len,)
    """
    hann_size = 4 * int(np.round(sample_rate / (kappa * delta))) + 1
    window = hann(hann_size)
    weighting_function = window[:int(np.ceil(hann_size / 2))]
    weighting_function = \
        np.pad(weighting_function,
               (toa_direct, rir_len - len(weighting_function) - toa_direct),
               constant_values=(0, 1))
    return weighting_function


def calc_scaling(rir_ism, raw_stochastic, weighting_function, target_drr,
                 upper_lim, lower_lim):
    """
    Calculate the scaling to adjust power of the stochastic part so that the
    resulting RIR will match the target DRR according to the equation
    target_drr = e_early / e_late, with e_early and e_late being the energy of
    the early reflections and late reverberation respectively, that are
    functions of the scaling to be determined (see eq. (6) in the paper).

    Args:
        rir_ism (array-like):
            RIR consisting of early reflections

        raw_stochastic (array-like):
            Stochastically modeled late reverberation of the RIR

        weighting_function (array_like):
            Weighting function for fade-in of the stochastic part

        target_drr (float):
            Target DRR of the RIR to be generated

        upper_lim (int):
            Index referring to the upper sum limit used for DRR calculations

        lower_lim (int):
            Index referring to the lower sum limit used for DRR calculations

    Returns:
        - None, if no solution to match the target DRR can be found
        - Scaling factor for stochastic part
    """
    energy_stochastic = (weighting_function * raw_stochastic) ** 2
    energy_ism = rir_ism ** 2
    component_product = rir_ism * weighting_function * raw_stochastic

    denominator = (target_drr * np.sum(energy_stochastic[upper_lim:])
                   - np.sum(energy_stochastic[lower_lim:upper_lim]))
    linear_term = (2 / denominator
                   * (target_drr * np.sum(component_product[upper_lim:])
                      - np.sum(component_product[lower_lim:upper_lim])))
    constant_term = 1 / denominator * (
            target_drr * np.sum(energy_ism[upper_lim:]) - np.sum(
                energy_ism[lower_lim:upper_lim]))

    root_arg = (linear_term / 2) ** 2 - constant_term
    if root_arg < 0:
        # not solvable for given values
        return None
    # consider one solution
    scaling = -linear_term / 2 + np.sqrt(root_arg)
    return scaling


def calc_directivity_attenuation(delta_pos, azimuth_directivity,
                                 colatitude_directivity, src_directivity):
    """
    Calculate attenuation caused by rotations of the sources.
    Take into account that the direct path component of the RIR is scaled by
    the source’s directional response, which is influenced by azimuth and
    elevation angles between the look direction of the source and the
    microphone position.

    Args:
        delta_pos (array-like):
            Position difference between microphone and source position.

        azimuth_directivity (float):
            Azimuth angle of the source seen from pyroomacoustics x- to y-axis.

        colatitude_directivity (float):
            Elevation angle of the source seen from pyroomacoustics
            z-axis to xy-plane.

        src_directivity:
            Directivity pattern of the source possible choices:
            'cardioid', 'hypercardioid', 'supercardioid' 'subcardioid' and
            'omnidirectional'

    Returns:
     Attenuation factor caused by the directivity pattern.
    """
    # calculate source azimuth and elevation in degree
    # for direct path between source and microphone
    azimuth = np.arctan2(delta_pos[1], delta_pos[0])
    azimuth *= 180 / np.pi
    colatitude = np.arctan2(np.linalg.norm(delta_pos[:2]), delta_pos[2])
    colatitude *= 180 / np.pi

    # get effective offset angles
    src_azimuth = azimuth - azimuth_directivity
    src_colatitude = colatitude - colatitude_directivity
    # add 90 degrees because pyroomacoustics assumes 90 degree as reference for
    # maximum directivity for later directivity response calculation
    src_colatitude += 90

    # create reference directivity pattern
    ref_directivity_src = CardioidFamily(
        orientation=DirectionVector(
            azimuth=0,
            colatitude=90,
            degrees=True
        ),
        pattern_enum=pattern_dispatcher[src_directivity]
    )
    # get attenuation w.r.t. reference characteristic at relative offset angles
    # of the source-microphone constellation
    directivity_attenuation = (
        ref_directivity_src.get_response(
            [src_azimuth, ], [src_colatitude, ]).item())
    return directivity_attenuation


def _add_stochastic_rir(
        rir_ism, rir_len, kappa, alpha, beta, distance, room_dimensions,
        t60, directivity_attenuation, win_len_direct, sample_rate,
        sound_speed, adapt_alpha, lower_limit_alpha
):
    """
    Create room impulse response consisting of early reflections
    simulated with image source method and stochastic reverberation.
    A target DRR is calculated and the individual parts consisting of ISM part,
    weighting function and exponentially decaying stochastic part are combined
    and appropriately scaled to match the target DRR.

    Args:
        rir_ism (array-like):
            Room impulse response (RIR) consisting of early reflections

        alpha (float):
            Directivity factor of acoustic source

        distance (float):
            Distance between microphone position and acoustic source

        directivity_attenuation (float):
            Source’s directional response based on source directivity,
            azimuth and elevation of the source

        adapt_alpha (bool):
            If true, maximum directivity factor based on early reflections
            is calculated to guarantee solvability

        lower_limit_alpha (float):
            Lower limit of the range where alpha factors are drawn from

        All other arguments are described in "calc_rirs".

    Returns:
        rir: None, if no solution could be found for given parameters
        otherwise combined and scaled RIR
        alpha: the directivity factor of acoustic source used for generation
    """
    # Maximum peak of the RIR generated by Pyroomacoustics is shifted by 40
    # samples compared to physical time of arrival (TOA)
    pra_toa_offset = 40

    # calculate stochastic reverberation part of RIR
    n = np.arange(rir_len)
    delta = 3 * np.log(10) / t60
    toa_direct = int(
        np.round(distance / sound_speed * sample_rate)) + pra_toa_offset
    raw_stochastic = np.random.normal(size=rir_len) * np.exp(
        -delta * (n - toa_direct) / sample_rate)
    raw_stochastic[:toa_direct + 1] = 0

    lower_lim = toa_direct - win_len_direct
    upper_lim = toa_direct + win_len_direct + 1

    # Adjust alpha to maintain solvability :
    # In some cases, the DRR of the RIR created by the image source method may
    # be already too low for the currently drawn alpha. Adding additional
    # reverberant parts typically decrease the DRR further, which causes an
    # invalid solution. If adapt_alpha is set, the value of alpha is adapted
    # based on the early reflections of the RIR so that a valid solution for
    # the scaling of the stochastic part of the RIR is guaranteed.
    if adapt_alpha:
        # calculate DRR of ISM part
        drr_im = (np.sum(rir_ism[lower_lim:upper_lim] ** 2)
                  / np.sum(rir_ism[upper_lim:] ** 2))
        # solve DRR formula to get alpha that would just create the
        # DRR of the early reflections of the RIR, seen as upper bound for the
        # maximum DRR to achieve, because only first reflections are simulated
        # and amending of the stochastic part would overall lower the DRR
        max_factor = (drr_im * np.pi * t60 * distance ** 2
                      / np.prod(room_dimensions) * 100
                      / directivity_attenuation ** 2)
        if max_factor < alpha:
            # For the current value of alpha, the target DRR would be larger
            # than the DRR of only the early reflections, so re-draw alpha,
            # that then creates a lower, but achievable target DRR and
            # therefore a solution during later scaling can be found
            if max_factor < lower_limit_alpha:
                # use omnidirectional directivity factor (1) as lower limit
                # because otherwise no solution can be found
                alpha = np.random.uniform(1, max_factor)
            else:
                # sample alpha so that it still lays within the original range,
                # but preserves convergence
                alpha = np.random.uniform(lower_limit_alpha, max_factor)

    target_drr = calc_target_drr(room_dimensions, alpha, beta, t60,
                                 directivity_attenuation, distance)
    weighting_function =\
        calc_window(delta, kappa, toa_direct, rir_len, sample_rate)

    scaling = calc_scaling(rir_ism, raw_stochastic, weighting_function,
                           target_drr, upper_lim, lower_lim)
    if scaling is None:
        return None, alpha
    else:
        rir = rir_ism + raw_stochastic * weighting_function * scaling
        return rir, alpha


def calc_rirs(
        src_pos, sensor_pos, room_dimensions, t60, azimuth_directivity,
        colatitude_directivity, src_directivity,
        directivity_factor_range=(2.5, 5.5), kappa=1, beta=1., rir_len=16384,
        max_order_ism=3, win_len_direct=40, sound_speed=343, sample_rate=16000,
        alpha_diff=0.1, max_retries=10
):
    """
    Room impulse response (RIR) simulation for microphone pairs, consisting of
    closely spaced microphones, as described in 'DIMINISHING DOMAIN MISMATCH
    FOR DNN-BASED ACOUSTIC DISTANCE ESTIMATION VIA STOCHASTIC ROOM
    REVERBERATION MODELS'

    Args:
        src_pos (array-like):
            Coordinates of acoustic source with shape (dimension,)

        sensor_pos (array-like):
            Coordinates of the two microphones of the microphone pair with
            shape (dimension, 2). It is assumed that the microphones have a
            reasonably spacing for the assumptions to hold.

        room_dimensions (array-like):
            Size of the room with shape (dimension)

        t60 (float):
            T60 time of the room in seconds

        azimuth_directivity (float):
            Azimuth of source in degrees

        colatitude_directivity (float):
            Colatitude of source in degrees

        src_directivity (str):
            Directivity pattern of the source; possible choices:
            'cardioid', 'hypercardioid', 'supercardioid' 'subcardioid' and
            'omnidirectional'

        directivity_factor_range (array-like):
            Limits of the uniform distribution from which the directivity
            factor of the source (alpha) is drawn

        beta (float):
            Directivity factor of microphone

        kappa (float):
            Determines the duration of the fade-in of the stochastic part of
            the RIR. Lower values cause smoother transition.

        rir_len (int):
            Length of RIR to be generated, in samples

        max_order_ism (int):
            Maximum order of simulated image sources

        win_len_direct (int):
            One-sided length of the window defining the part of the RIR
            which belongs to the direct path.

        sound_speed (int):
            Speed of sound used to determine the position of the direct path

        sample_rate (int):
            Sample rate

        alpha_diff (float):
            Maximum difference between alpha factors used for RIRs of different
            microphones to be tolerated.

        max_retries (int):
            Number of iterations to re-draw alpha factor;
            If no valid solution for the scaling of the stochastic part of the
            RIR could be found to maintain the target DRR before the maximum
            value of iteration is reached, the interval from which alpha is
            drawn is adapted to the setup at hand.

    Returns:
        Array of generated RIRs of shape (n_microphones, rir_len)
    """
    msg = "Invalid source directivity specified"
    assert src_directivity in pattern_dispatcher.keys(), msg

    msg = "Only simulation of microphone pairs is supported"
    assert len(np.array(sensor_pos).T) == 2, msg

    if np.linalg.norm(np.diff(sensor_pos, axis=1)) > 0.1:
        warnings.warn("The first microphone determines the directivity factor \
            for the other. This assumption may not hold for larger microphone \
            spacings where larger differences for alpha may be possible.")

    # simulate first "max_order_ism" reflections with pyroomacoustics
    e_absorption, _ = pra.inverse_sabine(t60, room_dimensions)
    room = pra.ShoeBox(
        room_dimensions, fs=sample_rate, materials=pra.Material(e_absorption),
        max_order=max_order_ism, use_rand_ism=True, max_rand_disp=0.001)

    directivity_src = CardioidFamily(
        orientation=DirectionVector(azimuth=azimuth_directivity,
                                    colatitude=colatitude_directivity,
                                    degrees=True),
        pattern_enum=pattern_dispatcher[src_directivity]
    )
    room.add_source(src_pos, directivity=directivity_src)
    room.add_microphone(sensor_pos)
    room.compute_rir()

    # Apply high-pass filter
    rirs_ism = np.zeros((2, rir_len))
    for ch_id, rir_ism in enumerate(room.rir):
        assert np.any(rir_ism != 0)
        rir_ism = high_pass(rir_ism[0])
        rirs_ism[ch_id, :len(rir_ism)] = rir_ism

    it_cnt = 0
    any_result_none = False
    adapt_alpha = False
    rirs_stochastic = np.zeros((2, rir_len))
    alpha_values = np.zeros(2)

    # calculate stochastic part until possible solution found
    while (np.abs(alpha_values[0] - alpha_values[1]) > alpha_diff
           or not all(np.any(rirs_stochastic, axis=-1))):
        if it_cnt == 0:
            alpha = np.random.uniform(*directivity_factor_range)
        elif any_result_none and it_cnt < max_retries:
            # rejection sampling for 'max_retries' iterations
            alpha = np.random.uniform(*directivity_factor_range)
        else:
            # if maximum possible factor calculated, choose minimum
            # to have same factor for the different microphones
            alpha = min(alpha_values)

        it_cnt += 1
        if it_cnt > max_retries:
            adapt_alpha = True

        for ch, rir_ism in enumerate(rirs_ism):
            # calculate directivity attenuation
            delta_pos = sensor_pos[:, ch] - np.asarray(src_pos)
            directivity_attenuation =\
                calc_directivity_attenuation(delta_pos, azimuth_directivity,
                                             colatitude_directivity,
                                             src_directivity)

            mic_pos = sensor_pos.T[ch]
            distance = np.linalg.norm(mic_pos - src_pos)

            # Note: the first microphone determines the directivity factor
            # since the directivity factor should be approximately the same for
            # both microphones if the inter-microphone distance is assumed to
            # be small. For larger inter-microphone distance, the directivity
            # factor might be different for both microphones.
            
            rir_result, alpha = _add_stochastic_rir(
                rir_ism, rir_len, kappa, alpha, beta, distance,
                room_dimensions, t60, directivity_attenuation, win_len_direct,
                sample_rate, sound_speed, adapt_alpha,
                directivity_factor_range[0])

            alpha_values[ch] = alpha
            if rir_result is None:
                any_result_none = True
                # continue to calculate the other possible values of alpha
                continue
            rirs_stochastic[ch] = rir_result[:rir_len]

    return rirs_stochastic
