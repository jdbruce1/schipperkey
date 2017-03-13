import numpy as np
from pitch_identification import identify_pitches_binned #, identify_pitches
from math import sqrt
from librosa import load

cmaj = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
cmin = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)

offset_map = {"C": 0, "Db": 1, "D": 2, "Eb": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "Ab": 8, "A": 9, "Bb": 10, "B": 11}
reverse_map = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A","Bb", "B"]


def remove_duplicates(lst):
    '''Removes duplicates from a list'''
    output = []
    for x in lst:
        if x not in output:
            output.append(x)
    return output

def check_relative(key1, key2):
    '''Jacob: I don't really know what this does'''
    if len(key1) >= 2 and key1[-1] == 'm':
        return reverse_map[(offset_map[key1[:-1]] + 3) % 12] == key2 or key1[:-1] == key2
    else:
        if len(key2) >= 2 and key2[-1] == 'm':
            return reverse_map[(offset_map[key2[:-1]] + 3) % 12] == key1 or key2[:-1] == key1
        else:
            return False

def compare_key_krumhansl(test, keyvector):
    '''Compares two vectors to give a similarity, based on paper by krumhansl
    inputs:     a test vector representing the total duration of pitches in the song
                a key vector representing the common prevalance of pitches in the key
    outputs:    a score for how similar the key is to the song'''
    x = test
    y = keyvector
    xavg = np.mean(x)
    yavg = np.mean(y)
    score = np.sum((x-xavg) * (y-yavg)) / sqrt(np.sum((x-xavg)**2)*np.sum((y-yavg)**2))
    return score

def get_rolled_totals(bins, offset, bins_per_pitchclass):
    bins = np.roll(bins, offset)
    vector = [np.sum(bins[i:bins_per_pitchclass+i]) for i in range(0, len(bins), bins_per_pitchclass)]
    # print vector
    return np.array(vector)

def get_bin_score(key_vector, offset, bins, bins_per_pitchclass):
    '''Wrapper for compare_key_krumhansl - takes the inputs from the caller and
    converts them to the inputs that compare_key_krumhansl expects'''
    return compare_key_krumhansl(get_rolled_totals(bins, offset, bins_per_pitchclass), np.array(key_vector))

def get_key_binned(path, name, method='yinfft', sr=22050):
    '''Gets the key from a file, using the binning method
    inputs:     a path to the file containing the melody (string)
                the name of the song (string)
                the method for the pitch tracker to use (fcomb for whistling, yinfft otherwise) (string)
                the sample rate for the file (number)
    outputs:    a (name list) tuple where the list is the top 5 best keys identified'''

    # if method == 'fcomb':
    #     print name, "is whistled"

    bins_per_pitchclass = 1
    bin_intensities = np.array(identify_pitches_binned(path, bins_per_pitchclass, method, sr=sr, disp=False))
    bin_intensities_shift = np.array(identify_pitches_binned(path, bins_per_pitchclass, method, pitchshift=.5, sr=sr, disp=False))
    key_likelihoods = np.zeros((len(bin_intensities), 4))

    for offset_index in range(len(bin_intensities)):
        key_likelihoods[offset_index,0] = get_bin_score(cmaj, offset_index, bin_intensities, bins_per_pitchclass)
        key_likelihoods[offset_index,1] = get_bin_score(cmin, offset_index, bin_intensities, bins_per_pitchclass)
        key_likelihoods[offset_index,2] = get_bin_score(cmaj, offset_index, bin_intensities_shift, bins_per_pitchclass)
        key_likelihoods[offset_index,3] = get_bin_score(cmin, offset_index, bin_intensities_shift, bins_per_pitchclass)

    offsets = np.repeat(np.arange(12*bins_per_pitchclass),4)
    modes = np.tile(np.array([0,1,2,3]), 12*bins_per_pitchclass)

    key_outputs = []

    for i in range(len(offsets)):
        if modes[i] % 2 == 1:
            best_mode = 'm'
        else:
            best_mode = ''
        pitchclass_index = (12 - offsets[i]/bins_per_pitchclass) % 12
        pitch_offset = (bins_per_pitchclass - offsets[i]) % bins_per_pitchclass -bins_per_pitchclass/2
        key_outputs.append([reverse_map[pitchclass_index], best_mode, key_likelihoods[offsets[i], modes[i]], pitch_offset ])
        # Also guess the key above this key if you're really sharp
        if modes[i] >= 2:
            key_outputs.append([reverse_map[(pitchclass_index+1)%12], best_mode, key_likelihoods[offsets[i], modes[i]], pitch_offset ])

    key_outputs = sorted(key_outputs, key=lambda x: x[2],reverse=True)

    key_outputs = [str(x[0]) +x[1] for x in key_outputs] #+'. Score: '+str(x[2])

    key_outputs = remove_duplicates(key_outputs)

    return (name, key_outputs[:5])
