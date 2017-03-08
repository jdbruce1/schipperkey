import sys
import os
import csv
import sounddevice
from librosa import load
import numpy as np
from key_identification import get_key_from_path, check_relative, get_key_binned


def get_keys(waves, labels=None):
    if labels is not None:
        return [get_key_from_file(wave, os.path.basename(wave), labels[os.path.basename(wave)][2]) for wave in waves] #[get_key(wave, wave) for wave in waves]
    else:
        return [get_key_from_file(wave, os.path.basename(wave)) for wave in waves]


# def get_key_from_file(filename, name, method="hummed"):
#     melody, sr = load(filename)
#     return get_key(melody, name, method=get_pitch_tracker_from_method(method), sr=sr)


def get_pitch_tracker_from_method(method):
    if method.lower()[0] == 'w':
        return 'fcomb'
    return 'yinfft'

def add_note(key, file):
    return load(file)


def score_keys(algorithm, correct):
    return [(score_key(algorithm[key], correct[key][0]), algorithm[key], correct[key][0], key) for key in algorithm.keys()]


def score_key(assigned_keys, correct_key):
    if assigned_keys[0] == correct_key:
        return 100
    total = 0
    try:
        index = assigned_keys.index(correct_key)
        total += 40 + 10 * (len(assigned_keys) - 1 - index)
    except StandardError:
        pass

    if check_relative(assigned_keys[0], correct_key):
        total += 10

    return total


def record(sr):
    max_frames = sr*60

    recording = sounddevice.rec(frames=max_frames, samplerate=sr, channels=1)
    custom_input("(Press Enter to stop recording.)")
    sounddevice.stop()

    return np.trim_zeros(recording)


def read_label_file(in_filename):
    in_file = open(in_filename, 'rb')
    reader = csv.reader(in_file, delimiter=",")
    pairs = [(line[0] + ".wav", line[1:]) for line in reader]
    return pairs


def test_mode():
    folder = custom_input("Enter the path to a folder with .wav files that you would like to know the key of: ")

    while not os.path.isdir(folder):
        folder = custom_input("Folder could not be found.  Please input the path to the desired folder: ")
    label_file = custom_input("Enter the path to a file that stores the correct keys for the input folder (Enter \"skip\" to continue without labels.): ")
    while label_file != "skip" and (not os.path.isfile(label_file) or os.path.splitext(label_file)[1] != ".csv"):
        label_file = custom_input("Could not find the file specified. (Make sure it is in .csv format.)  Try again: ")
    if label_file == "skip":
        labels = None
    else:
        labels = dict(read_label_file(label_file))

    waves = [folder + "/" + path for path in os.listdir(folder) if os.path.splitext(path)[1] == ".wav"]
    output_keys = dict(get_keys(waves, labels))
    if labels is not None:
        scores = score_keys(output_keys, labels)
        output = scores
    else:
        output = [(line[0][0], line[1]) for line in output_keys]

    total_score = 0
    num_zeros = 0
    for melody in output:
        if len(melody) is 4:
            print "Assigned Keys:", melody[1]
            print "Correct Key:", melody[2]
            print "Score:", melody[0]
            total_score += melody[0]
            if melody[0] == 0:
                num_zeros += 1
            print "Filename:", melody[3]
        else:
            print "Assigned Key:", melody[0]
            print "Filename:", melody[1]
        print

    print "Average Score:", 1.0*total_score/len(output)
    print "Number of zero scores:", num_zeros
    print

    return 0


def demo_mode():
    choice = custom_input("Will you record now (R) or read from file (F)? ")
    while choice not in ['f', 'F', 'file', 'File', 'r', 'R', 'record', 'Record']:
        choice = custom_input("Input not understood.  Record now (R) or read from file (F)? ")
    if choice in ['r', 'R', 'record', 'Record']:
        response = 'y'
        while response in ['y', 'Y', 'yes', 'Yes']:
            custom_input("Press Enter to begin recording your melody.")
            sr = 22050
            recording = record(sr)
            custom_input("Press Enter to play the recording back.")
            sounddevice.play(recording, sr)
            custom_input("(Press Enter to stop playback.)")
            sounddevice.stop()
            response = custom_input("Would you like to record a different melody or move on? (Y to record again.): ")
        melody = recording
        name = "New recording"
    else:
        file = custom_input("Enter the path to a file to get the key of: ")
        while not os.path.isfile(file) or os.path.splitext(file)[1] != ".wav":
            file = custom_input("File could not be found. (Make sure it is .wav format.) Try again: ")
        melody, sr = load(file)
        name = file

    method = custom_input("How was this melody performed? (Whistled (w), Hummed (h), Sung (s), or Other (o)?")
    while method not in ['w', 'W', 'whistled', 'Whistled', 'h', 'H', 'hummed', 'Hummed', 's', 'S', 'sung', 'Sung', 'o', 'O', 'other', 'Other']:
        method = custom_input("I did not recognize that method.  Please input again: (w, h, s, or o?")

    key = get_key_binned(melody, name, method=get_pitch_tracker_from_method(method), sr=sr) #get_key(melody, name)
    print "Assigned Keys:"
    for i in range(len(key[1])):
        print "\t", str(i+1) + ".", key[1][i]

    print

    response = 'y'
    while response in ['y', 'Y', 'yes', 'Yes']:
        chosen_key = custom_input("Pick one of those keys to hear under your melody. (Enter \"skip\" to skip this step.): ")
        while chosen_key != "skip" and chosen_key not in key[1]:
            chosen_key = custom_input("Did not recognize input.  Please pick one of the assigned keys: ")
        if chosen_key == "skip":
            break
        note_added, sr = add_note(chosen_key, file)
        sounddevice.play(note_added, sr)
        custom_input("(Press Enter to stop.)")
        sounddevice.stop()
        response = custom_input("Would you like to hear another key? (Y for yes, no otherwise): ")
    response = custom_input("Would you like to work on another recording? (Y for yes, no otherwise): ")
    if response in ['y', 'Y', 'yes', 'Yes']:
        return 1
    else:
        return 0


def custom_input(prompt=""):
    print prompt
    out = raw_input()
    if out is 'q' or out is 'quit' or out is 'exit':
        exit(0)
    return out


if len(sys.argv) > 1 and sys.argv[1].strip() != '-t':
    print "Usage: schipperkey.py [-t]"
    exit(0)

if len(sys.argv) > 1:
    output = test_mode()
else:
    output = demo_mode()
    while output is 1:
        output = demo_mode()

# print get_key('toy_data/Twinkle.wav')
