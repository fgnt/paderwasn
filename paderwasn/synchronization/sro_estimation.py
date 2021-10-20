import numpy as np
from paderbox.transform import STFT

from paderwasn.synchronization.time_shift_estimation import max_time_lag_search


class OnlineWACD:
    def __init__(self,
                 seg_len=16384,
                 seg_shift=2048,
                 frame_shift_welch=1024,
                 fft_size=8192,
                 temp_dist=16384,
                 eps_max=400,
                 avg_period=160,
                 k_min=100,
                 k_max=1800):
        """Online weighted average coherence drift (WACD) method

        WACD-based online sampling rate offset (SRO) estimator from "Online
        Estimation of Sampling Rate Offset in Wireless Acoustic Sensor Networks
        with Packet Loss"

        Args:
            seg_len (int):
                Length of the segments used for coherence estimation (= Length
                of the segments used for power spectral density (PSD)
                estimation based on a Welch method)
            seg_shift (int):
                Shift of the segments used for coherence estimation (The SRO is
                estimated every seg_shift samples)
            frame_shift_welch (int):
                Frame shift used for the Welch method utilized for
                PSD estimation
            fft_size (int):
                Frame size / FFT size used for the Welch method utilized for
                PSD estimation
            temp_dist (int):
                Amount of samples between the two consecutive coherence
                functions
            eps_max (float):
                Maximum expected SRO in parts-per-million (ppm)
            avg_period (int):
                The average coherence product is averaged over the last
                avg_period calculated coherence products before SRO estimation
                (Needed in dynamic scenarios with time-varying SROs or source
                position changes whereby moving sources cannot be handled). If
                avg_period is None (default), the average coherence product up
                to the current point in time will be used for coherence
                estimation (Suitable for static scenarios with constant SROs
                and fixed positions of the  microphones and acoustic source).
            k_min (int):
                Lower limit of the frequency bin indices to be used for
                SRO estimation
            k_max (int):
                Upper limit of the  frequency bin indices to be used for
                SRO estimation
        """
        self.seg_len = seg_len
        self.seg_shift = seg_shift
        self.frame_shift_welch = frame_shift_welch
        self.fft_size = fft_size
        self.temp_dist = temp_dist
        self.eps_max = eps_max * 1e-6
        self.stft = STFT(shift=frame_shift_welch, size=fft_size,
                         window_length=fft_size, pad=False, fading=False)
        if k_min is not None:
            self.k_min = k_min
        else:
            self.k_min = 1
        if k_max is not None:
            # Limit the considered frequency range so that the normalized phase
            # does not suffer from phase wrap around effects, i.e., the maximum
            # frequency index is determined based on eps_max if k_max is larger
            # than N / (2 * eps_max * temp_dist)
            self.k_max = np.minimum(
                int(fft_size / (2 * self.eps_max * temp_dist)), k_max
            )
        else:
            self.k_max = int(fft_size / (2 * self.eps_max * temp_dist))
        self.avg_period = avg_period

    def __call__(self, sig, ref_sig):
        """Estimate the SRO of the single channel signal sig w.r.t. the single
        channel reference signal ref_sig

        Args:
            sig (array-like):
                Vector corresponding to the signal whose SRO should be
                estimated
            ref_sig (array-like):
                Vector corresponding to the reference signal (Should have the
                same length as sig)

        Returns:
            sro_estimates (numpy.ndarray):
                Vector containing the SRO estimates in ppm
        """
        # Maximum number of segments w.r.t. the reference signal (The actual
        # number of segments might be smaller due to the compensation of the
        # SRO-induced signal shift)
        num_segments = int(
            (len(ref_sig) - self.temp_dist - self.seg_len + self.seg_shift)
            // self.seg_shift
        )
        sro_estimates = np.zeros(num_segments)

        # The SRO-induced signal shift will be estimated based on the
        # SRO estimates
        tau_sro = 0

        # If avg_period is None, the average coherence product up to the
        # current point in time will be used for coherence estimation.
        # Otherwise, the average of the last avg_period coherence products will
        # be utilized.
        if self.avg_period is None:
            avg_coh_prod = \
                np.zeros(self.k_max - self.k_min, dtype=np.complex128)
        else:
            coherence_products = np.zeros(
                (self.avg_period, self.k_max - self.k_min), dtype=np.complex128
            )

        k = np.arange(self.k_min, self.k_max)
        # Estimate the SRO segment-wisely every seg_shift samples
        for seg_idx in range(num_segments):
            #  Estimate of the SRO-induced integer time shift to be compensated
            shift = int(np.round(tau_sro))

            # Calculate the coherence Gamma(seg_idx*seg_shift,k). Note that
            # the segment taken from sig starts at (seg_idx*seg_shift+shift)
            # in order to coarsely compensate the SRO induced delay.
            start = seg_idx * self.seg_shift + self.temp_dist
            seg_ref = ref_sig[start:start+self.seg_len]
            seg = sig[start+shift:start+shift+self.seg_len]
            coherence = self.calc_coherence(seg, seg_ref)

            # Calculate the coherence Gamma(seg_idx*seg_shift-temp_dist,k).
            # Note that the segment taken from sig starts at
            # (seg_idx*seg_shift-temp_dist+shift) in order to coarsely
            # compensate the SRO induced delay.
            start_delayed = seg_idx * self.seg_shift
            seg_ref_delayed = ref_sig[start_delayed:start_delayed+self.seg_len]
            seg_delayed = \
                sig[start_delayed+shift:start_delayed+shift+self.seg_len]
            coherence_delayed = \
                self.calc_coherence(seg_delayed, seg_ref_delayed)

            # Calculate the complex conjugated product of consecutive
            # coherence functions for the considered frequency range
            coherence_product = \
                (coherence * np.conj(coherence_delayed))[self.k_min:self.k_max]

            # Average the product of consecutive coherence functions over time
            if self.avg_period is None:
                # Calculate the moving average over all available
                # coherence products.
                avg_coh_prod = ((avg_coh_prod * seg_idx + coherence_product)
                                / (seg_idx + 1))
            else:
                # Calculate the avergae coherence product from the last
                # avg_period coherence products.
                coherence_products = np.roll(coherence_products, -1, 0)
                coherence_products[-1] = coherence_product
                avg_coh_prod = np.mean(coherence_products, 0)

            # Normalize the phase of the coherence product and project it into
            # the complex plane. Afterwards, calculate the weighted average
            # over all frequency bins and derive the SRO estimate.
            norm_phase = (np.angle(avg_coh_prod) * self.fft_size
                          / (2 * self.temp_dist * k * self.eps_max))
            mean_phase = \
                np.mean(np.abs(avg_coh_prod) * np.exp(1j * norm_phase))
            sro_est = self.eps_max / np.pi * np.angle(mean_phase)
            sro_estimates[seg_idx] = sro_est

            # Use the current SRO estimate to update the estimate for the
            # SRO-induced time shift between the signal and the reference
            # signal(The SRO-induced shift corresponds to the average shift of
            # the segment w.r.t. the center of the segment).
            if seg_idx == 0:
                # The center of the first segment is given by
                # (seg_len / 2 + temp_dist).
                tau_sro = (self.seg_len / 2 + self.temp_dist) * sro_est
            else:
                # The center of the other segments is given by
                # (seg_len / 2 + temp_dist + seg_idx * seg_shift)
                tau_sro += self.seg_shift * sro_est

            # If the end of the next segment from sig is larger than the length
            # of sig stop SRO estimation.
            nxt_end = ((seg_idx + 1) * self.seg_shift
                       + self.temp_dist + int(np.round(tau_sro))
                       + self.seg_len)
            if nxt_end > len(sig):
                return sro_estimates[:seg_idx + 1] * 1e6
        return sro_estimates * 1e6

    def calc_psd(self, seg_i, seg_j):
        """Estimate the (cross) power spectral density (PSD) from the given
        signal segments using a Welch method
        
        Args:
            seg_i (array-like):
                Vector with seg_len elements corresponding to the segment taken
                from the i-th signal
            seg_j (array-like):
                Vector with seg_len elements corresponding to the segment taken
                from the j-th signal
        Returns:
            psd_ij (numpy.ndarray):
                PSD of the the i-th signal and the j-th signal
        """
        psd_ij = np.mean(self.stft(seg_i) * np.conj(self.stft(seg_j)), axis=0)
        return psd_ij

    def calc_coherence(self, seg, seg_ref):
        """Estimate the coherence from the given signal segments

        Args:
            seg (array-like):
                Vector with seg_len elements corresponding to the segment taken
                from signal whose SRO should be estimated
            seg_ref (array-like):
                Vector with seg_len elements corresponding to the segment taken
                from the reference signal
        Returns:
            gamma (numpy.ndarray):
                Coherence of the signal and the reference signal
        """
        cpsd = self.calc_psd(seg_ref, seg)
        psd_ref_sig = self.calc_psd(seg_ref, seg_ref)
        psd_sig = self.calc_psd(seg, seg)
        gamma = cpsd / (np.sqrt(psd_ref_sig * psd_sig) + 1e-13)
        return gamma


class DynamicWACD:
    def __init__(self,
                 seg_len=8192,
                 seg_shift=2048,
                 frame_shift_welch=512,
                 fft_size=4096,
                 temp_dist=8192,
                 alpha=.95,
                 src_activity_th=.75,
                 settling_time=40):
        """Dynamic weighted average coherence drift (DWACD) method

        Sampling rate offset (SRO) estimator for dynamic scenarios with
        time-varying SROs and position changes of the acoustic source from
        "On Synchronization of Wireless Acoustic Sensor Networks in the
        presence of Time-Varying Sampling Rate Offsets and Speaker Changes"
        (Note that moving sources cannot be handled)

        Args:
            seg_len (int):
                Length of the segments used for coherence estimation (= Length
                of the segments used for power spectral density (PSD)
                estimation based on a Welch method)
            seg_shift (int):
                Shift of the segments used for coherence estimation (The SRO is
                estimated every seg_shift samples)
            frame_shift_welch (int):
                Frame shift used for the Welch method utilized for
                PSD estimation
            fft_size (int):
                Frame size / FFT size used for the Welch method utilized for
                PSD estimation
            temp_dist (int):
                Amount of samples between the two consecutive coherence
                functions
            alpha (float):
                Smoothing factor used for the autoregressive smoothing for time
                averaging of the complex conjugated coherence product
            src_activity_th (float):
                If the amount of time with source activity within one segment
                is smaller than the threshold src_activity_th the segment will
                not be used to update th average coherence product.
            settling_time (int):
                Amount of segments after which the SRO is estimated for the
                first time
        """
        self.seg_len = seg_len
        self.seg_shift = seg_shift
        self.frame_shift_welch = frame_shift_welch
        self.fft_size = fft_size
        self.temp_dist = temp_dist
        self.stft = STFT(shift=frame_shift_welch, size=fft_size,
                         window_length=fft_size, pad=False, fading=False)
        self.src_activity_th = src_activity_th * self.seg_len
        self.settling_time = settling_time
        self.alpha = alpha

    def __call__(self, sig, ref_sig, activity_sig, activity_ref_sig):
        """Estimate the SRO of the single channel signal sig w.r.t. the single
        channel reference signal ref_sig

        Args:
            sig (array-like):
                Vector corresponding to the signal whose SRO should be
                estimated
            ref_sig (array-like):
                Vector corresponding to the reference signal (Should have the
                same length as sig)
            activity_sig(array-like):
                Vector containing the sample-wise information of source
                activity in the signal sig
            activity_ref_sig(array-like):
                Vector containing the sample-wise information of source
                activity in the reference signal ref_sig
        Returns:
            sro_estimates (numpy.ndarray):
                Vector containing the SRO estimates in ppm
        """
        # Maximum number of segments w.r.t. the reference signal (The actual
        # number of segments might be smaller due to the compensation of the
        # SRO-induced signal shift)
        num_segments = int(
            (len(ref_sig) - self.temp_dist - self.seg_len + self.seg_shift)
            // self.seg_shift
        )
        sro_estimates = np.zeros(num_segments)
        avg_coh_prod = np.zeros(self.fft_size)

        # The SRO-induced signal shift will be estimated based on the
        # SRO estimates
        tau_sro = 0

        sro_est = 0
        # Estimate the SRO segment-wisely every seg_shift samples
        for seg_idx in range(num_segments):
            # Estimate of the SRO-induced integer shift to be compensated
            shift = int(np.round(tau_sro))

            # Check if an acoustic source is active in all signal segments
            # needed to calculate the current product of complex conjugated
            # coherence functions. The average coherence product is only
            # updated if an acoustic source is active for at least
            # src_activity_th * seg_len samples within each considered signal
            # segment.
            start = seg_idx * self.seg_shift + self.temp_dist
            activity_seg = \
                activity_sig[start+shift:start+shift+self.seg_len]
            activity_seg_ref = activity_ref_sig[start:start+self.seg_len]
            start_delayed = seg_idx * self.seg_shift
            activity_seg_delayed = \
                activity_sig[start_delayed+shift:start+shift+self.seg_len]
            activity_seg_ref_delayed = \
                activity_ref_sig[start_delayed:start_delayed+self.seg_len]
            activity = (
                    np.sum(activity_seg_ref_delayed) > self.src_activity_th
                    and np.sum(activity_seg_ref) > self.src_activity_th
                    and np.sum(activity_seg_delayed) > self.src_activity_th
                    and np.sum(activity_seg) > self.src_activity_th
            )

            if activity:
                # Calculate the coherence Gamma(seg_idx*seg_shift,k). Note
                # that the segment taken from sig starts at
                # (seg_idx*seg_shift+shift) in order to coarsely compensate
                # the SRO induced delay.
                start = seg_idx * self.seg_shift + self.temp_dist
                seg_ref = ref_sig[start:start+self.seg_len]
                seg = sig[start+shift:start+shift+self.seg_len]
                coherence = self.calc_coherence(seg, seg_ref, sro_est)

                # Calculate the coherence Gamma(seg_idx*seg_shift-temp_dist,k).
                # Note that the segment taken from sig starts at
                # (seg_idx*seg_shift-temp_dist+shift) in order to coarsely
                # compensate the SRO induced delay.
                start_delayed = seg_idx * self.seg_shift
                seg_ref_delayed = \
                    ref_sig[start_delayed:start_delayed+self.seg_len]
                seg_delayed = \
                    sig[start_delayed+shift:start_delayed+shift+self.seg_len]
                coherence_delayed = \
                    self.calc_coherence(seg_delayed, seg_ref_delayed, sro_est)

                # Calculate the complex conjugated product of consecutive
                # coherence functions for the considered frequency range
                coherence_product = coherence * np.conj(coherence_delayed)

                # Note that the used STFT exploits the symmetry of the FFT of
                # real valued input signals and computes only the non-negative
                # frequency terms. Therefore, the negative frequency terms
                # have to be added.
                coherence_product = np.concatenate(
                    [coherence_product[:-1],
                     np.conj(coherence_product)[::-1][:-1]],
                    -1
                )

                # Update the average coherence product
                avg_coh_prod = (self.alpha * avg_coh_prod
                                + (1 - self.alpha) * coherence_product)

                # Interpret the coherence product as generalized cross power
                # spectral density, use an efficient golden section search
                # to find the time lag which maximizes the corresponding
                # generalized cross correlation and derive the SRO from the
                # time lag.
                sro_est = - max_time_lag_search(avg_coh_prod) / self.temp_dist
            if seg_idx > self.settling_time - 1:
                sro_estimates[seg_idx] = sro_est
            if seg_idx == self.settling_time - 1:
                sro_estimates[:seg_idx + 1] = sro_est

            # Use the current SRO estimate to update the estimate for the
            # SRO-induced time shift (The SRO-induced shift corresponds to
            # the average shift of the segment w.r.t. the center of the
            # segment).
            if seg_idx == self.settling_time - 1:
                # The center of the first segment is given by
                # (seg_len / 2 + temp_dist). The center of the other segments
                # is given by (seg_len / 2 + temp_dist + seg_idx * seg_shift).
                tau_sro += (.5 * self.seg_len + self.temp_dist) * sro_est
                tau_sro += self.seg_shift * sro_est * (self.settling_time - 1)
            elif seg_idx >= self.settling_time:
                # The center of the other segments is given by
                # (seg_len / 2 + temp_dist + seg_idx * seg_shift)
                tau_sro += self.seg_shift * sro_est

            # If the end of the next segment from sig is larger than the length
            # of sig stop SRO estimation.
            nxt_end = ((seg_idx + 1) * self.seg_shift
                       + self.temp_dist + int(np.round(tau_sro))
                       + self.seg_len)
            if nxt_end > len(sig):
                return sro_estimates[:seg_idx + 1] * 1e6
        return sro_estimates * 1e6

    def calc_psd(self, seg_i, seg_j, sro=0.):
        """Estimate the (cross) power spectral density (PSD) from the given
        signal segments using a Welch method

        Args:
            seg_i (array-like):
                Vector with seg_len elements corresponding to the segment taken
                from the i-th signal
            seg_j (array-like):
                Vector with seg_len elements corresponding to the segment taken
                from the j-th signal
            sro (float):
                SRO to be compensated in the Welch method
        Returns:
            psd_ij (numpy.ndarray):
                PSD of the the i-th signal and the j-th signal
        """
        stft_seg_j = self.stft(seg_j)
        shifts = sro * self.frame_shift_welch * np.arange(len(stft_seg_j))
        stft_seg_j *= \
            np.exp(1j * 2 * np.pi / self.fft_size
                   * np.arange(self.fft_size // 2 + 1)[None] * shifts[:, None])
        return np.mean(self.stft(seg_i) * np.conj(stft_seg_j), axis=0)

    def calc_coherence(self, seg, seg_ref, sro):
        """Estimate the coherence from the given signal segments

        Args:
            seg (array-like):
                Vector with seg_len elements corresponding to the segment taken
                from signal whose SRO should be estimated
            seg_ref (array-like):
                Vector with seg_len elements corresponding to the segment taken
                from the reference signal
            sro (float):
                SRO to be compensated when calculating the PSDs needed for
                coherence estimation
        Returns:
            gamma (numpy.ndarray):
                Coherence of the signal and the reference signal
        """
        cpsd = self.calc_psd(seg_ref, seg, sro)
        psd_ref_sig = self.calc_psd(seg_ref, seg_ref)
        psd_sig = self.calc_psd(seg, seg)
        gamma = cpsd / (np.sqrt(psd_ref_sig * psd_sig) + 1e-13)
        return gamma
