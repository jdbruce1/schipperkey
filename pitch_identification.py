import sys
import numpy as np
import os.path
import matplotlib.pyplot as plt

def get_piano_keys():
    piano_freqs = []#np.zeros(88)
    piano_freqs.append(27.5)
    factor = 2.0**(1.0/12)
    for i in range(1, 88):
        piano_freqs.append(piano_freqs[i-1] * factor)
    piano_freqs = [round(x,2) for x in piano_freqs]
    octave = ["A","A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
    piano_notes = octave * 7 + octave[:4]

    # print piano_freqs
    # print piano_notes
    # print len(piano_freqs)
    # print len(piano_notes)
    return np.array(piano_freqs), piano_notes

def snap_to_pitchclass(pitches):
    piano_freqs, piano_notes = get_piano_keys()

    matched_pitches = np.zeros_like(pitches)
    matched_notes = []

    for pitch_index in range(len(pitches)):
        # print pitch_index
        tracked_pitch = pitches[pitch_index]
        # print tracked_pitch

        if (tracked_pitch < min(piano_freqs)) or (tracked_pitch > max(piano_freqs)):
            matched_pitches[pitch_index] = 0
            # print matched_pitches[pitch_index]
        else:
            piano_index = np.argmin(abs(piano_freqs-pitches[pitch_index]))
            # print piano_index

            matched_pitches[pitch_index] = piano_freqs[piano_index]#min(piano_freqs, key=lambda x:abs(x-pitches[pitch_index]))
            # print matched_pitches[pitch_index]

            matched_notes.append(piano_notes[piano_index])
            # print matched_notes[-1]
        print

    return matched_notes

def aggregate_pitchclass(matched_notes):
    agg_notes = {}
    for matched_note in matched_notes:
        try:
            agg_notes[matched_note] += 1
        except KeyError:
            agg_notes[matched_note] = 1
    return agg_notes

def vectorize(agg_notes):
    vector = [] # 0 index corresponds to C

    for note_name in ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#","A","A#", "B"]:
        try:
            vector.append(agg_notes[note_name])
        except KeyError:
            vector.append(0)

    return vector

def moving_average(pitches, window=5, display=False):
    # takes a moving average of the pitches to reduce variance

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


def debounce(pitches, tolerance = 1.5, display=False):
    # debounces by removing big jumps in pitch
    # tolerance should be greater than 1
    # larger tolerance allows more variance
    output = []

    for index in range(1, pitches.size):
        last_pitch = pitches[index-1]
        this_pitch = pitches[index]
        if (last_pitch is 0) and (not this_pitch is 0):
            # we're leaving from zero, include this pitch
            output.append(this_pitch)
        elif (this_pitch < last_pitch*tolerance) and (this_pitch > last_pitch/tolerance):
            # there's not a huge change, include this pitch
            output.append(this_pitch)

    output = np.array(output)

    if display:
        plt.figure()
        plt.plot(output)
        plt.show()
    return output


# def post_process(pitches):
#     # takes a list of pitches and processes them so as to reduce effects

def pitch_track(path, sr=22050, downsample=1, win_size=4096, hop_size=512, tolerance=.8, display=False):
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

def identify_pitches(path):

    pitches = pitch_track(path, display=True)
    # pitches = debounce(pitches, tolerance=1.2, display=True)
    matched_notes = snap_to_pitchclass(pitches)

    # print matched_notes

    agg_notes = aggregate_pitchclass(matched_notes)
    print agg_notes
    return vectorize(agg_notes)

print identify_pitches('toy_data/Silent.wav')
