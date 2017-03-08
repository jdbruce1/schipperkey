import numpy as np
from pitch_identification import identify_pitches_binned
from math import sqrt
from librosa import load

cmaj = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
cmin = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)
cmaj_norm = [x/sum(cmaj) for x in cmaj]
cmin_norm = [x/sum(cmin) for x in cmin]
offset_map = {"C": 0, "Db": 1, "D": 2, "Eb": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "Ab": 8, "A": 9, "Bb": 10, "B": 11}
reverse_map = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A","Bb", "B"]


def check_relative(key1, key2):
    if len(key1) >= 2 and key1[-1] == 'm':
        return reverse_map[(offset_map[key1[:-1]] + 3) % 12] == key2 or key1[:-1] == key2
    else:
        if len(key2) >= 2 and key2[-1] == 'm':
            return reverse_map[(offset_map[key2[:-1]] + 3) % 12] == key1 or key2[:-1] == key1
        else:
            return False

# def get_key_from_path(path, name, method='yinfft', sr=22050):
#     # the end method requires a path, so we should pass a path when we can
#     # we should try to move toward this method
#
#     pitch_intensities = np.array(identify_pitches_from_path(path, method, sr=sr))
#
#     key_likelihoods = np.zeros((len(pitch_intensities), 2))
#     print pitch_intensities
#
#     for offset_index in range(12):
#         #key_likelihoods[offset_index,0] = np.dot(pitch_intensities, get_key_vector_ind(offset_index, 'major'))
#         #key_likelihoods[offset_index,1] = np.dot(pitch_intensities, get_key_vector_ind(offset_index, 'minor'))
#         key_likelihoods[offset_index,0] = compare_key_krumhansl(pitch_intensities, get_key_vector_ind(offset_index, 'major'))
#         key_likelihoods[offset_index,1] = compare_key_krumhansl(pitch_intensities, get_key_vector_ind(offset_index, 'minor'))
#
#     # print key_likelihoods
#
#     threshold = -1
#     print key_likelihoods
#     best_indices = np.where(key_likelihoods > threshold) #np.unravel_index(np.argmax(key_likelihoods), key_likelihoods.shape)
#     print best_indices
#     best_offsets = best_indices[0]
#     best_modes = best_indices[1]
#
#     return_stuff = []
#
#     for i in range(len(best_offsets)):
#         if best_modes[i] == 1:
#             best_mode = 'm'
#         else:
#             best_mode = ''
#         return_stuff.append([reverse_map[best_offsets[i]], best_mode, key_likelihoods[best_offsets[i], best_modes[i]]])
#
#     return_stuff = sorted(return_stuff, key=lambda x: x[2],reverse=True)
#
#     return_stuff = [str(x[0]) +x[1] for x in return_stuff] #+'. Score: '+str(x[2])
#
#     return (name, return_stuff[:5])

# def get_key(melody, name, method='yinfft', sr=22050):
#     # gets the key from a melody
#     # takes a signal, its sample rate, and its name
#     # returns the name and a list of top key options in sorted order
#     # melody, sr = load(path)
#
#     pitch_intensities = np.array(identify_pitches(melody, method, sr=sr))
#
#     key_likelihoods = np.zeros((len(pitch_intensities), 2))
#     print pitch_intensities
#
#     for offset_index in range(12):
#         #key_likelihoods[offset_index,0] = np.dot(pitch_intensities, get_key_vector_ind(offset_index, 'major'))
#         #key_likelihoods[offset_index,1] = np.dot(pitch_intensities, get_key_vector_ind(offset_index, 'minor'))
#         key_likelihoods[offset_index,0] = compare_key_krumhansl(pitch_intensities, get_key_vector_ind(offset_index, 'major'))
#         key_likelihoods[offset_index,1] = compare_key_krumhansl(pitch_intensities, get_key_vector_ind(offset_index, 'minor'))
#
#     # print key_likelihoods
#
#     threshold = -1
#     print key_likelihoods
#     best_indices = np.where(key_likelihoods > threshold) #np.unravel_index(np.argmax(key_likelihoods), key_likelihoods.shape)
#     print best_indices
#     best_offsets = best_indices[0]
#     best_modes = best_indices[1]
#
#     return_stuff = []
#
#     for i in range(len(best_offsets)):
#         if best_modes[i] == 1:
#             best_mode = 'm'
#         else:
#             best_mode = ''
#         return_stuff.append([reverse_map[best_offsets[i]], best_mode, key_likelihoods[best_offsets[i], best_modes[i]]])
#
#     return_stuff = sorted(return_stuff, key=lambda x: x[2],reverse=True)
#
#     return_stuff = [str(x[0]) +x[1] for x in return_stuff] #+'. Score: '+str(x[2])
#
#     return (name, return_stuff[:5])

def compare_key_krumhansl(test, keyvector):
    x = test
    y = keyvector
    xavg = np.mean(x)
    yavg = np.mean(y)
    score = np.sum((x-xavg) * (y-yavg)) / sqrt(np.sum((x-xavg)**2)*np.sum((y-yavg)**2))
    return score

def get_key_vector(note, mode):
    if mode == "major":
        return roll(cmaj, offset_map[note])
    elif mode == "minor":
        return roll(cmin, offset_map[note])
    return None

def get_key_vector_ind(index, mode):
    # as get_key_vector but using pitchlass indices, rather than pitchclasses
    if mode == "major":
        return roll(cmaj, index)
    elif mode == "minor":
        return roll(cmin, index)
    return None

def roll(vector, offset):
    if offset >= len(vector):
        raise ValueError("Offset must be less than vector length.")
    end = vector[len(vector)-offset:] + vector[:len(vector)-offset]
    if offset == 8:
        print end
    return end


def get_rolled_totals(bins, offset, bins_per_pitchclass):
    bins = np.roll(bins, offset)
    vector = [np.sum(bins[i:bins_per_pitchclass+i]) for i in range(0, len(bins), bins_per_pitchclass)]
    # print vector
    return np.array(vector)


def get_bin_score(key_vector, offset, bins, bins_per_pitchclass):
    return compare_key_krumhansl(get_rolled_totals(bins, offset, bins_per_pitchclass), np.array(key_vector))


def get_key_binned(path, name, method='yinfft', sr=22050):
    # gets the key from a path
    # takes a signal, its sample rate, and its name
    # returns the name and a list of top key options in sorted order

    # melody, sr = load(path)
    bins_per_pitchclass = 1
    bin_intensities = np.array(identify_pitches_binned(path, bins_per_pitchclass, method, sr=sr))
    # bin_intensities = [5, 40, 2, 17, 0, 57, 52, 8, 66, 10, 35, 2]
    # bin_intensities = np.repeat(bin_intensities, bins_per_pitchclass)
    # print "Bin intensities: ", bin_intensities
    key_likelihoods = np.zeros((len(bin_intensities), 2))

    for offset_index in range(len(bin_intensities)):
        #key_likelihoods[offset_index,0] = np.dot(pitch_intensities, get_key_vector_ind(offset_index, 'major'))
        #key_likelihoods[offset_index,1] = np.dot(pitch_intensities, get_key_vector_ind(offset_index, 'minor'))
        key_likelihoods[offset_index,0] = get_bin_score(cmaj, offset_index, bin_intensities, bins_per_pitchclass)
        key_likelihoods[offset_index,1] = get_bin_score(cmin, offset_index, bin_intensities, bins_per_pitchclass)

    # print key_likelihoods

    threshold = -1
    best_indices = np.where(key_likelihoods > threshold) #np.unravel_index(np.argmax(key_likelihoods), key_likelihoods.shape)
    best_offsets = best_indices[0]
    best_modes = best_indices[1]

    return_stuff = []

    for i in range(len(best_offsets)):
        if best_modes[i] == 1:
            best_mode = 'm'
        else:
            best_mode = ''
        pitchclass_index = (12 - best_offsets[i]/bins_per_pitchclass) % 12
        return_stuff.append([reverse_map[pitchclass_index], best_mode, key_likelihoods[best_offsets[i], best_modes[i]], best_offsets[i]%bins_per_pitchclass-bins_per_pitchclass/2])

    return_stuff = sorted(return_stuff, key=lambda x: x[2],reverse=True)

    return_stuff = [str(x[0]) +x[1] + str(x[3]) for x in return_stuff] #+'. Score: '+str(x[2])

    return (name, return_stuff[:5])
