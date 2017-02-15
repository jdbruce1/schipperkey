# file to play around with some packages

import numpy as np, scipy as sp, librosa, cmath,math
# from IPython.display import Audio

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

# def round_to_pitch(frequency):


twinkle, sample_rate = librosa.load("twinkle.wav")

piano_freqs, piano_notes = get_piano_keys()

pitches, magnitudes = librosa.core.piptrack(twinkle, sr=sample_rate, fmin=150.0, fmax=4000.0, threshold=0.1)

print pitches.shape
print magnitudes.shape

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

print matched_notes


# Twinkle Twinkle Melody in C
# CC GG AA G  FF EE DD C

# print max_pitches

# wavwrite('output.wav', twinkle, sample_rate)

# Audio(twinkle, rate = sample_rate)

# print sample_rate
