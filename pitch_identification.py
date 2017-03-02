import sys
import numpy as np
import os.path
from os import remove
import matplotlib.pyplot as plt
import scipy as sp

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

def get_piano_keys():
    # computes paired lists of the frequencies of a piano and their corresponding pitch classes
    # returns: np array of piano frequencies, list of piano pitch classes
    piano_freqs = []#np.zeros(88)
    piano_freqs = 440 * 2**(np.arange(-48, 40)/12.0)
    piano_freqs = [round(x,2) for x in piano_freqs]
    octave = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
    piano_notes = octave * 7 + octave[:4]

    # print piano_freqs
    # print piano_notes
    # print len(piano_freqs)
    # print len(piano_notes)
    return np.array(piano_freqs), piano_notes

def snap_to_pitchclass(pitches):
    # snaps input pitches into bins by pitch class
    # input: pitch sequence from a pitch tracker [261.625565, 261.625565, 261.625565, 293.65]
    # output: pitchclass sequence, like ["C","C","C","D"]
    piano_freqs, piano_notes = get_piano_keys()

    matched_pitches = np.zeros_like(pitches)
    matched_notes = []

    for pitch_index in range(len(pitches)):
        # print pitch_index
        tracked_pitch = pitches[pitch_index]
        # print tracked_pitch

        # feels weird to make pitch 0 when greater than maximum frequency, but I guess it's kind of an error code
        if (tracked_pitch < min(piano_freqs)) or (tracked_pitch > max(piano_freqs)):
            matched_pitches[pitch_index] = 0
            # print matched_pitches[pitch_index]
        else:
            # we should consider doing a log-scale comparison of nearness
            piano_index = np.argmin(abs(piano_freqs-pitches[pitch_index]))
            # print piano_index

            matched_pitches[pitch_index] = piano_freqs[piano_index]#min(piano_freqs, key=lambda x:abs(x-pitches[pitch_index]))
            # print matched_pitches[pitch_index]

            matched_notes.append(piano_notes[piano_index])
            # print matched_notes[-1]

    return matched_notes

def aggregate_pitchclass(matched_notes):
    # counts the instances of each pitchclass
    # input: a list of pitchlasses, like ["C","C","C", "D"]
    # output: a dictionary of the pitchlasses keyed to counts, like {"C":500, "D":200}
    agg_notes = {}
    for matched_note in matched_notes:
        try:
            agg_notes[matched_note] += 1
        except KeyError:
            agg_notes[matched_note] = 1
    return agg_notes

def vectorize(agg_notes):
    # turns a dictionary of note names keyed to intensities into a vector
    # input: agg_notes, like {"C":500, "D":200}
    # ouput: pitch vector, like [500, 0, 200, ...]
    vector = [] # 0 index corresponds to C

    for note_name in ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#","A","A#", "B"]:
        try:
            vector.append(agg_notes[note_name])
        except KeyError:
            vector.append(0)

    return vector

def moving_average(pitches, window=5, display=False):
    # takes a moving average of the pitches to reduce variance
    # window is double sided .. total will be averaged over 2*window values
    output = np.zeros_like(pitches)

    for index in range(window, pitches.size-window):
        output[index] = np.mean(pitches[index-window:index+window])

    if display:
        plt.figure()
        plt.plot(output)
        plt.show()
    return output

# def remove_jumps(pitches):
#
#     output = []
#     for index in range(1, pitches.size):
#         last_pitch = pitches[index-1]
#         this_pitch = pitches[index]


# def debounce(pitches, tolerance = 1.5, display=False):
#     # debounces by removing big jumps in pitch
#     # tolerance should be greater than 1
#     # larger tolerance allows more variance
#
#     #I'm not sure what I think about
#     # 1. this deletes pitches, rather than moving them
#     # 2. note jumps might not meet the tolerance.  Maybe we should be looking for quick up AND down
#     # or vice versa, rather than just a one-way change.  We're looking for spikes, not consistent
#     # pitch change.
#     output = []
#
#     for index in range(1, pitches.size):
#         last_pitch = pitches[index-1]
#         this_pitch = pitches[index]
#         if (last_pitch is 0) and (not this_pitch is 0):
#             # we're leaving from zero, include this pitch
#             output.append(this_pitch)
#         elif (this_pitch < last_pitch*tolerance) and (this_pitch > last_pitch/tolerance):
#             # there's not a huge change, include this pitch
#             output.append(this_pitch)
#
#     output = np.array(output)
#
#     if display:
#         plt.figure()
#         plt.plot(output)
#         plt.show()
#     return output


# def post_process(pitches):
#     # takes a list of pitches and processes them so as to reduce effects

def pitch_track(path, sr=22050, downsample=1, win_size=4096, hop_size=512, tolerance=.8, display=False):
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

    pitch_o = pitch("yin", win_s, hop_s, samplerate)
    pitch_o.set_unit("midi")
    pitch_o.set_tolerance(tolerance)

    pitches = []
    confidences = []

    # total number of frames read
    total_frames = 0
    reading = True
    while reading:
        samples, read = s()
        pitch = pitch_o(samples)[0]
        #pitch = int(round(pitch))
        confidence = pitch_o.get_confidence()
        # if confidence < 0.8: pitch = 0.
        # print("%f %f %f" % (total_frames / float(samplerate), pitch, confidence))
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

def identify_pitches(signal, sr=22050):
    # analyzes a signal into a vector of pitch intensities
    # hacked together temporary method
    # returns: a vector, as from identify_pitches_from_path
    temp_path = 'temp.wav'

    wavwrite(temp_path, signal, sr)
    output = identify_pitches_from_path(temp_path, sr)
    remove(temp_path)

    return output


def identify_pitches_from_path(path, sr=22050):
    # analyzes a wav file into a vector of pitch intensities
    # input: path to a wav file
    # returns: a vector of pitch intensities, starting at C

    pitches = pitch_track(path, sr=sr, display=False)
    # pitches = debounce(pitches, tolerance=1.2, display=True)
    matched_notes = snap_to_pitchclass(pitches)

    agg_notes = aggregate_pitchclass(matched_notes)
    return vectorize(agg_notes)

# print identify_pitches('toy_data/Twinkle.wav')