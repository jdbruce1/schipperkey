# file to play around with some packages

import numpy as np, scipy as sp, librosa, cmath,math
import time
import fluidsynth
import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic
from matplotlib import pyplot as plt
from librosa import load
import librosa
from pitch_identification import identify_pitches, pitch_track, remove_jumps, bin_pitches, get_note_onsets, average_note_pitch
# import pitch_identification

def get_piano_keys():
    piano_freqs = []#np.zeros(88)
    piano_freqs.append(27.5)
    factor = 2.0**(1.0/12)
    for i in range(1, 88):
        piano_freqs.append(piano_freqs[i-1] * factor)
    octave = ["A","A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
    piano_notes = octave * 7 + octave[:4]

    # print piano_freqs
    # print piano_notes
    # print len(piano_freqs)
    # print len(piano_notes)
    return piano_freqs, piano_notes

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

def test_librosa(filepath):
    signal, sample_rate = librosa.load(filepath)
    pitches, magnitudes = librosa.core.piptrack(signal, sr=sample_rate, fmin=150.0, fmax=4000.0, threshold=0.1)
    print "Pitches size: ", pitches.shape
    print "Magnitudes size: ", magnitudes.shape
    num_times = pitches.shape[1]

    max_pitches = np.zeros(num_times)
    for time_frame_index in range(num_times):
        max_ind = np.argmax(magnitudes[:,time_frame_index])
        max_pitches[time_frame_index] = pitches[max_ind, time_frame_index]

    plt.plot(max_pitches, label='whatever', color='green')
    plt.show()

def extract_pitch_sequence(filepath):
    signal, sample_rate = librosa.load(filepath)
    piano_freqs, piano_notes = get_piano_keys()
    pitches, magnitudes = librosa.core.piptrack(signal, sr=sample_rate, fmin=150.0, fmax=4000.0, threshold=0.1)
    num_times = pitches.shape[1]
    num_freqs = pitches.shape[0]

    max_pitches = np.zeros(num_times)
    for time_frame_index in range(num_times):
        max_ind = np.argmax(magnitudes[:,time_frame_index])
        max_pitches[time_frame_index] = pitches[max_ind, time_frame_index]

    matched_pitches = np.zeros_like(max_pitches)
    matched_notes = []

    for pitch_index in range(len(max_pitches)):
        # print max_pitches[index]
        matched_pitches[pitch_index] = min(piano_freqs, key=lambda x:abs(x-max_pitches[pitch_index]))
        # print matched_pitches[index]
        matched_notes.append(piano_notes[piano_freqs.index(matched_pitches[pitch_index])])

    return matched_notes

# print extract_pitch_sequence("twinkle.wav")
# Twinkle Twinkle Melody in C
# CC GG AA G  FF EE DD C



# wavwrite('output.wav', twinkle, sample_rate)

def test_fluid_synth():

    fs = fluidsynth.Synth()

    fs.start()

    sfid = fs.sfload("example.sf2")
    fs.program_select(0, sfid, 0, 0)

    fs.noteon(0, 60, 30)
    fs.noteon(0, 67, 30)
    fs.noteon(0, 76, 30)

    time.sleep(1.0)

    fs.noteoff(0, 60)
    fs.noteoff(0, 67)
    fs.noteoff(0, 76)

    time.sleep(1.0)

    fs.delete()

def test_pYAAPT(filepath):
    signal = basic.SignalObj(filepath)
    pitch = pYAAPT.yaapt(signal, **{'f0_min': 27.5, 'frame_length' : 50.0, 'frame_space' : 25.0, 'f0_max': 4186.0})
    print "Pitch shape: ", pitch.values.shape
    plt.plot(pitch.values, label='pchip interpolation', color='green')
    plt.show()

def test_chromagram():
    cmaj_sig, sr = load('toy_data/cmaj_sung.wav')
    # print cmaj_sig
    chrom_vect = identify_pitches_chromagram(cmaj_sig)

    print chrom_vect

def test_rj(path):
    pitches = pitch_track(path, display=True)
    # pitches = remove_jumps(pitches, 30, 50)
    #
    # plt.figure()
    # plt.plot(pitches)
    # plt.show()

    pitches = remove_jumps(pitches, 5, 50, display=True)

    # matched_notes = snap_to_pitchclass(pitches)
    #
    # agg_notes = aggregate_pitchclass(matched_notes)
    # return vectorize(agg_notes)

def test_whistle_id(path):
    mfcc = librosa.cqt(load(path)[0])
    plt.figure()
    librosa.display.specshow(mfcc)
    plt.show()

def test_bins(path):
    pitches = pitch_track(path, method='yinfft', sr=22050,display=False)
    pitches = remove_jumps(pitches, 15, 200, display=False)
    pitches = remove_jumps(pitches, 5, 200, display = False)
    onsets = get_note_onsets(pitches)
    pitches = average_note_pitch(pitches, onsets, display=False)

    print bin_pitches(pitches, 8)

# whistled examples

test_bins('toy_data/twinkle.wav')
