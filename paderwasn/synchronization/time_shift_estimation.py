import numpy as np
from paderbox.array.segment import segment_axis


from paderwasn.synchronization.utils import golden_section_max_search


def max_time_lag_search(gcpsd):
    """Search for the time lag which maximizes the generalized cross
    correlation (GCC) corresponding to the given generalized cross power
    spectral density (GCPSD)

    Args:
        gcpsd(array_like):
            Vector corresponding to a GCPSD
    Returns:
        lambda_max (float):
            Time lag which maximizes the GCC function
    """

    def _eval_gcc(gcpsd, lag):
        """Evaluate the GCC corresponding to the given GCPSD at the given
        non-integer time lag

        Args:
            gcpsd (array-like):
                Vector corresponding to the GCPSD
            lag (float):
                Non-integer time lag for which the GCC function has to
                be evaluated
        Returns:
            gcc (float):
                Value of the GCC function at the given time lag
        """
        gcpsd = np.asarray(gcpsd)
        fft_size = len(gcpsd)
        k = np.fft.fftshift(
            np.arange(-fft_size // 2, fft_size // 2))
        pre_factor = 1j * 2 * np.pi / fft_size * k
        gcc = np.abs(np.sum(gcpsd * np.exp(pre_factor * lag)))
        return gcc

    gcc = np.fft.ifftshift(np.real(np.fft.ifft(gcpsd)))
    lambda_max = np.argmax(gcc) - len(gcpsd) // 2
    search_interval = (lambda_max - .5, lambda_max + .5)
    lambda_max = golden_section_max_search(
        lambda x: _eval_gcc(gcpsd, x), search_interval
    )
    return lambda_max


def est_time_shift(sig, ref_sig, seg_size, seg_shift):
    """Estimate the time shift between two signals

    The time shift is estimated based on the generalized cross correlation with
    phase transform (GCC-PhaT).

    Args:
        sig (numpy.ndarray):
            Vector corresponding to a signal
        ref_sig (numpy.ndarray):
            Vector corresponding to the signal being used as reference
        seg_size (int):
            Size of the segments used in the GCC-PhaT algorithm
        seg_shift:
            Shift of the segments used in the GCC-PhaT algorithm
    Returns:
        shifts (numpy.ndarray):
            Vector corresponding to the estimated time shifts
    """
    def _get_gcpsd(seg, seg_ref):
        """Calculate the generalized cross power spectral density (GCPSD) for
        the given signal segments
        
        Args:
            seg (array-like):
                Vector corresponding to a segment of a signal
            seg_ref (array-like):
                Vector corresponding to the segment of the reference signal
        Returns:
            gcpsd (numpy.ndarray):
                Vector corresponding to the GCPSD
        """
        fft_seg = np.fft.fft(seg)
        fft_ref_seg = np.fft.fft(seg_ref)
        cpsd = np.conj(fft_ref_seg) * fft_seg
        gcpsd = cpsd / (np.abs(fft_seg) * np.abs(fft_ref_seg) + 1e-18)
        return gcpsd

    segments = segment_axis(sig, seg_size, seg_shift, end='cut')
    segments_ref = segment_axis(ref_sig, seg_size, seg_shift, end='cut')
    shifts = np.zeros(len(segments))
    for seg_idx, (seg, ref_seg) in enumerate(zip(segments, segments_ref)):
        shifts[seg_idx] = max_time_lag_search(_get_gcpsd(seg, ref_seg))
    return shifts
