import sys
import numpy as np
import os.path
from os import remove
import matplotlib.pyplot as plt
import scipy as sp
import librosa

def wavwrite(filepath, data, sr, norm=True, dtype='int16',):
    '''
    Write wave file using scipy.io.wavefile.write, converting from a float (-1.0 : 1.0) numpy array to an integer array

    Parameters
    ----------
    filepath : str
        The path of the output .wav file
    data : np.array
        The float-type audio array
    sr : int
        The sampling rate
    norm : bool
        If True, normalize the audio to -1.0 to 1.0 before converting integer
    dtype : str
        The output type. Typically leave this at the default of 'int16'.
    '''
    if norm:
        data /= np.max(np.abs(data))
    data = data * np.iinfo(dtype).max
    data = data.astype(dtype)
    sp.io.wavfile.write(filepath, sr, data)



def bin_pitches(pitches, bins_per_pitchclass):
    # Takes in a numpy array of frequencies in Hz, and returns a numpy
    # array of energies in frequency bins.  There are bins_per_pitchclass * 88
    # elements in the output array.  The value in each bin is equal to the
    # number of elements in the input array that fell within that bin's frequencies.
    top_piano_note =  440*2**(39.0/12)
    bottom_piano_note = 440*2**(-48.0/12)
    num_bins = 88 * bins_per_pitchclass
    bin_edges = bottom_piano_note * (2**(-.5/12)) * np.logspace(0, 88.0/12, num_bins+1, base=2)

    bin_indices = np.searchsorted(bin_edges, pitches) - 1

    bin_energies = np.zeros(num_bins)
    pitches[pitches > top_piano_note] = 0

    for pi in range(len(pitches)):
        bi = bin_indices[pi]
        if bi != -1 and not (bi >= len(bin_energies)):
            bin_energies[bi] += 1

    return bin_energies


def binclass(bin_energies, bins_per_pitchclass):
    # takes 88*bins_per_pitchclass bin_energies and maps to 12*bins_per_pitchclass bin_energies
    # kind of like a chromagram, where bins with the same log2 value are mapped to the same new bin

    oct_len = 12*bins_per_pitchclass

    octaves = np.zeros((8,oct_len))
    for oct_ind in range(8):
        try:
            octaves[oct_ind] = bin_energies[oct_ind*oct_len:(oct_ind+1)*oct_len]
        except ValueError:
            octaves[oct_ind,:len(bin_energies)-oct_ind*oct_len] = bin_energies[oct_ind*oct_len:]
            octaves[oct_ind,len(bin_energies)-oct_ind*oct_len:] = np.nan

    binclass = np.nansum(octaves, axis=0) / np.isfinite(octaves).sum(0) # normalize by number of notes counted
    return np.roll(binclass, -3*bins_per_pitchclass) # roll back to align c to 0 index (rather than A)


## Currently unused.
def net_change_in_window(vector, start, end):
    total = 0
    for i in range(start+1, end):
        total += np.abs(vector[i] - vector[i-1])
    return total

## Currently unused
def moving_average(pitches, window=10, threshold=.1, display=False):
    # takes a moving average of the pitches to reduce variance
    # window is double sided .. total will be averaged over 2*window values
    output = np.zeros_like(pitches)

    for index in range(window, pitches.size-window):
        # size is twice window
        avg_change = net_change_in_window(pitches, index-window, index+window)/2.0/window
        if avg_change/pitches[index] > threshold:
            continue

        output[index] = np.mean(pitches[index-window:index+window])

    if display:
        plt.figure()
        plt.plot(output)
        plt.show()
    return output


def get_note_onsets(pitches, lag=5):
    # Detects changes of notes by looking for sustained
    # change in pitch.  Returns an array of (onset, release) pairs.
    # lag: how many frames of stability required to count as a note.
    onsets = []
    in_note = False
    onset = 0
    for i in range(lag, len(pitches)):
        if pitches[i] == 0 and pitches[i-lag] == 0:
            continue
        if pitches[i] == 0 and pitches[i-lag] != 0:
            if in_note:
                onsets.append((onset, i-lag))
                in_note = False
            continue
        up_threshold = pitches[i] * 2**(.8/12)
        down_threshold = pitches[i] * 2**(-.8/12)
        if pitches[i-lag] > up_threshold or pitches[i-lag] < down_threshold:
            if in_note:
                onsets.append((onset, i-lag))
                in_note = False
        elif not in_note:
            onset = i-lag
            in_note = True
        if in_note and pitches[i-lag] == 0:
            print "in note when previous was zero", pitches[i], pitches[i-lag], i
    if pitches[-1] != 0:
        onsets.append((onset, len(pitches)))
    return onsets


def average_note_pitch(pitches, onsets, display=False):
    # Takes a sequence of pitches and detected note onset/release pairs, and
    # averages the pitch over those durations.
    output = np.zeros_like(pitches)
    for i in range(len(onsets)):
        output[onsets[i][0]:onsets[i][1]] = np.mean(pitches[onsets[i][0]:onsets[i][1]])

    if display:
        plt.figure()
        plt.plot(output)
        plt.show()
    return output


def remove_jumps(pitches, window_size, threshold, display=False):
    # removes spikes in pitch
    # if a pitch jumps more than threshold and jumps again before window_size, the jump is removed

    start_peak = 0
    for index in range(1, pitches.size):
        last_pitch = pitches[index-1]
        this_pitch = pitches[index]
        if abs(this_pitch - last_pitch) > threshold:
            # there's a big jump here
            if index - start_peak < window_size:
                pitches[start_peak:index] = 0
            else:
                start_peak = index

    if display:
        plt.figure()
        plt.plot(pitches)
        plt.show()

    return pitches


def pitch_track(path, method='yinfft', sr=22050, downsample=1, win_size=2048, hop_size=512, tolerance=.8, display=False):
    # uses aubio pitch tracker to turn a signal into a sequence of pitches
    # input: path to a wav file, and options
    # output: sequence of pitches, in Hz
    from aubio import source, pitch
    samplerate = sr // downsample
    if len( sys.argv ) > 2: samplerate = int(sys.argv[2])

    win_s = win_size // downsample # fft size
    hop_s = hop_size // downsample # hop size

    s = source(path, samplerate, hop_s)

    samplerate = s.samplerate

    pitch_o = pitch(method, win_s, hop_s, samplerate)
    pitch_o.set_tolerance(tolerance)
    pitch_o.set_silence(-35.0)

    pitches = []
    confidences = []

    # total number of frames read
    total_frames = 0
    reading = True
    while reading:
        samples, read = s()
        pitch = pitch_o(samples)[0]
        confidence = pitch_o.get_confidence()
        pitches += [pitch]
        confidences += [confidence]
        total_frames += read
        if read < hop_s:
            reading = False

    if 0: sys.exit(0)

    output = np.array(pitches)

    if display:
        plt.figure()
        plt.plot(output)
        plt.show()

    return output


def identify_pitches_binned(path, bins_per_pitchclass, method, sr=22050, disp=False):
    # Takes the path to a .wav file, and returns a vector of bins_per_pitchclass*12 elements
    # representing how strongly those frequencies appear in the input file.  Each bin corresponds
    # to one part of a western note, so at minimum, there should be 12 bins.
    # plt.ion()
    pitches = pitch_track(path, method=method, sr=sr,display=disp)
    pitches = remove_jumps(pitches, 15, 200, display=disp)
    pitches = remove_jumps(pitches, 5, 200, display = disp)
    onsets = get_note_onsets(pitches)
    pitches = average_note_pitch(pitches, onsets, display=disp)
    binned_pitches = bin_pitches(pitches, bins_per_pitchclass)
    bc = binclass(binned_pitches, bins_per_pitchclass)

    if disp:
        print "Binclass:", bc

    return bc
