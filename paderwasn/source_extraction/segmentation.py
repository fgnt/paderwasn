import numpy as np
import paderbox as pb


def get_activities_time_domain(
        activities, frame_size=1024, frame_shift=256, margin=1600
):
    """
    Derive time-domain activities from a frame-wise activity

    Args:
        activities (np.ndarray):
            Frame-wise activities of the sources (Shape: (number of sources x
            number of frames))
        frame_size (int):
            Frame size used to calculate the STFT.
        frame_shift (int):
            Frame shift used to calculate the STFT.
        margin:
            Amount of samples
    Returns:
        activities_time (np.ndarray):
            Sample-wise activities of the sources (Shape: (number of sources x
            number of samples))
    """
    activities_time = np.zeros(
        (len(activities), activities.shape[-1] * frame_shift + frame_size),
        bool
    )

    for i, activity in enumerate(activities):
        for segment in pb.array.interval.ArrayInterval(activity).intervals:
            frame_onset, frame_offset = segment
            onset = np.maximum(frame_onset * frame_shift - margin, 0)
            offset = frame_offset * frame_shift + margin

            activities_time[i, onset:offset] = True
    return activities_time


def bridge_pauses_transcription(activity, min_pause_len=16000):
    """
    Discard pauses within activity which are smaller than a given
    minimum length.

    Args:
        activity (np.ndarray):
            Activity of a source.
        min_pause_len:
            Minimum length of a pause. Activity pauses which are smaller than
             this value are discarded.
    Returns:
        Activity after discarding short pauses
    """
    on = []
    off = []
    for interval in pb.array.interval.ArrayInterval(activity).intervals:
        start, stop = interval
        on.append(start)
        off.append(stop)
    onsets = []
    offsets = []
    if len(on) > 0:
        onset = on[0]
        for i in range(len(on) - 1):
            current_offset = off[i]
            nxt_onset = on[i+1]
            if nxt_onset - current_offset > min_pause_len:
                onsets.append(onset)
                offsets.append(current_offset)
                onset = nxt_onset
        onsets.append(onset)
        offsets.append(off[-1])
        onsets = [np.maximum(0, onset) for onset in onsets]
        offsets = [offset for offset in offsets]
    activitiy_processed = np.zeros_like(activity, np.bool)
    for on, off in zip(onsets, offsets):
        activitiy_processed[on:off] = True
    return activitiy_processed


def segment_by_activity(sig, activity):
    """
    Segment a dignal into segments of contiguous activity. Sections of
    inactivity are discarded,

    Args:
        sig (np.ndarray):
            Signal to be segmented.
        activity (np.ndarray):
            Activity (boolean array) which is used to segment the signal.
    Returns:
        List of signal segments, list of segment lengths and list
        of segment onsets
    """
    segments = []
    num_samples = []
    onsets = []
    for segment in pb.array.interval.ArrayInterval(activity).intervals:
        onset, offset = segment
        segments.append(sig[onset:offset])
        num_samples.append(offset - onset)
        onsets.append(onset)
    return segments, num_samples, onsets
