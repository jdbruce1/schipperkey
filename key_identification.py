import numpy as np
from pitch_identification import identify_pitches
from math import sqrt
from librosa import load

cmaj = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
cmin = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)
cmaj_norm = [x/sum(cmaj) for x in cmaj]
cmin_norm = [x/sum(cmin) for x in cmin]
offset_map = {"C": 0, "Db": 1, "D": 2, "Eb": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "Ab": 8, "A": 9, "Bb": 10, "B": 11}
reverse_map = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#","A","A#", "B"]

def get_key_temp(melody, name, sr=22050):
    # gets the key from a melody
    # takes a signal, its sample rate, and its name
    # returns the name and a list of top key options in sorted order
    # melody, sr = load(path)

    pitch_intensities = np.array(identify_pitches(melody, sr))

    key_likelihoods = np.zeros((len(pitch_intensities), 2))
    print pitch_intensities

    for offset_index in range(12):
        # key_likelihoods[offset_index,0] = np.dot(pitch_intensities, get_key_vector_temp(offset_index, 'major'))
        # key_likelihoods[offset_index,1] = np.dot(pitch_intensities, get_key_vector_temp(offset_index, 'minor'))
        key_likelihoods[offset_index,0] = compare_key_krumhansl(pitch_intensities, get_key_vector_temp(offset_index, 'major'))
        key_likelihoods[offset_index,1] = compare_key_krumhansl(pitch_intensities, get_key_vector_temp(offset_index, 'minor'))

    # print key_likelihoods

    threshold = .9 * np.amax(key_likelihoods)
    print key_likelihoods
    best_indices = np.where(key_likelihoods > threshold) #np.unravel_index(np.argmax(key_likelihoods), key_likelihoods.shape)
    print best_indices
    best_offsets = best_indices[0]
    best_modes = best_indices[1]

    return_stuff = []

    for i in range(len(best_offsets)):
        if best_modes[i] == 1:
            best_mode = 'minor'
        else:
            best_mode = 'major'
        return_stuff.append([reverse_map[best_offsets[i]], best_mode, key_likelihoods[best_offsets[i], best_modes[i]]])

    return_stuff = sorted(return_stuff, key=lambda x: x[2],reverse=True)

    return_stuff = [str(x[0]) + ' '+x[1] for x in return_stuff] #+'. Score: '+str(x[2])

    return (name, return_stuff)

def compare_key_krumhansl(test, keyvector):
    x = test
    y = keyvector
    xavg = np.mean(x)
    yavg = np.mean(y)
    score = np.sum((x-xavg) * (y-yavg)) / sqrt(np.sum((x-xavg)**2)*np.sum((y-yavg)**2))
    print score
    return score

def get_key_vector(note, mode):
    if mode == "major":
        return roll(cmaj, offset_map[note])
    elif mode == "minor":
        return roll(cmin, offset_map[note])
    return None

def get_key_vector_temp(index, mode):
    # as get_key_vector but using pitchlass indices, rather than pitchclasses
    if mode == "major":
        return roll(cmaj, index)
    elif mode == "minor":
        return roll(cmin, index)
    return None

def roll(vector, offset):
    if offset >= len(vector):
        raise ValueError("Offset must be less than vector length.")
    return vector[offset:] + vector[:offset]
