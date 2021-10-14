import numpy as np
from scipy.signal.windows import hann


def compensate_sro(sig, sro, fft_size=8192):
    """Compensate the sampling rate offset (SRO) of the the given signal using
    the STFT resampling method introduced in "Efficient Sampling Rate Offset
    Compensation - An Overlap-Save Based Approach"

    The function can handle constant and time-varying SROs (SRO trajectories)

    Args:
        sig (array-like):
            Vector corresponding to the signal to be resampled
        sro (float or array-like):
            SRO in parts-per-million (ppm) to be compensated. If sro is a
            scalar value it will be treated as constant SRO. Otherwise, sro
            is interpreted as SRO trajectory corresponding to a time-varying
            SRO.
        fft_size (int):
            FFT size used for calculating the STFT. The STFT uses a frame size
            of fft_size/2 and a frame shift of fft_size/4.
    Returns:
        sig_resampled (numpy.ndarray):
            Vector corresponding to the resampled signal
    """
    if not np.isscalar(sro):
        sro = np.asarray(sro)
        max_block_idx = len(sro)
    else:
        max_block_idx = 0

    sro *= 1e-6
    k = np.fft.fftshift(np.arange(-fft_size / 2, fft_size / 2))
    block_len = int(fft_size // 2)
    sig_resamp = np.zeros_like(sig)
    shift_sro = 0
    block_idx = 0
    len_rest = 0

    # The STFT uses a Hann window with 50% overlap as analysis window
    win = hann(block_len, sym=False)

    while True:
        # Accumulate the SRO-induced signal shifts (The shifts correspond to
        # the average shift within the block)
        if np.isscalar(sro):
            shift_sro = sro * (block_idx * block_len / 2 + block_len / 2)
        else:
            shift_sro += sro[block_idx] * block_len / 2

        # Separate the SRO- induced shift into its integer and its
        # fractional part. The integer part is handled by a correposinding
        # shift of the analysis window. The remaining fractional part is
        # handled by a phase shift using a multiplication with a complex valued
        # exponential function.
        integer_shift = np.round(shift_sro)
        rest_shift = integer_shift - shift_sro

        # Compensate the integer part of the SRO-induced signal shift
        block_start = int(block_idx * block_len / 2 + integer_shift)
        if block_start < 0:
            if block_start + block_len < 0:
                block = np.zeros(block_len)
            else:
                block = np.pad(sig[0:block_start + block_len],
                               (block_len - (block_start + block_len), 0),
                               'constant')
        else:
            if (block_start + block_len) > sig.size:
                block = np.zeros(block_len)
                block[:sig[block_start:].size] = sig[block_start:]
                len_rest = sig[block_start:].size
            else:
                block = sig[block_start:block_start + block_len]

        # Compensate the fractional part of the SRO-induced signal shift
        sig_fft = np.fft.fft(win * block, fft_size)
        sig_fft *= np.exp(-1j * 2 * np.pi * k / fft_size * rest_shift)

        # Go back to the time domain and add the blocks with an overlap of 50%
        block_start = int(block_idx * block_len / 2)
        if block_start+block_len > sig_resamp.size:
            n_pad = block_start + block_len - sig_resamp.size
            sig_resamp = np.pad(sig_resamp, (0, n_pad), 'constant')
        sig_resamp[block_start:block_start + block_len] += \
            np.real(np.fft.ifft(sig_fft))[:block_len]

        # Stop resampling if the end of the signal or the sro trajectory
        # is reached
        block_end = int(block_idx * block_len / 2 - integer_shift) + block_len
        if block_end > sig.size or block_idx == max_block_idx - 1:
            return sig_resamp[:block_start+len_rest]
        block_idx += 1


def sim_sro(sig, sro, fft_size=8192):
    """Simulate the sampling rate offset (SRO) of the the given signal using
    the STFT resampling method introduced in "Efficient Sampling Rate Offset
    Compensation - An Overlap-Save Based Approach"

    The function can handle constant and time-varying SROs (SRO trajectories)

    Args:
        sig (array-like):
            Vector corresponding to the signal to be resampled
        sro (float or array-like):
            SRO in parts-per-million (ppm) to be simulated. If sro is a
            scalar value it will be treated as constant SRO. Otherwise, sro
            is interpreted as SRO trajectory corresponding to a time-varying
            SRO.
        fft_size (int):
            FFT size used for calculating the STFT. The STFT uses a frame size
            of fft_size/2 and a frame shift of fft_size/4.
    Returns:
        sig_resampled (numpy.ndarray):
            Vector corresponding to the resampled signal
    """
    return compensate_sro(sig, -sro, fft_size)
