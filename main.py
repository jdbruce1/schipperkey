import sys
import os
import csv
import sounddevice
from librosa import load
import numpy

def get_keys(waves):
    return [get_key(wave, wave) for wave in waves]


def get_key(wave, name):
    return name, ['C', 'D', 'E', 'F', 'G']


def add_note(key, file):
    return load(file)


def score_keys(algorithm, correct):
    return [(10, algorithm[key][0], correct[key][1], key) for key in algorithm.keys()]


def record(sr):
    max_frames = sr*60

    recording = sounddevice.rec(frames=max_frames, samplerate=sr, channels=1)
    custom_input("(Press any key to stop recording.)")
    sounddevice.stop()

    return numpy.trim_zeros(recording)


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

    waves = [path for path in os.listdir(folder) if os.path.splitext(path)[1] == ".wav"]
    output_keys = dict(get_keys(waves))
    if labels is not None:
        scores = score_keys(output_keys, labels)
        output = scores
    else:
        output = [(line[0][0], line[1]) for line in output_keys]

    for melody in output:
        if len(melody) is 4:
            print "Assigned Key:", melody[1]
            print "Correct Key:", melody[2]
            print "Score:", melody[0]
            print "Filename:", melody[3]
        else:
            print "Assigned Key:", melody[0]
            print "Filename:", melody[1]
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
            sr = 8000
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

    key = get_key(melody, name)
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
    out = raw_input(prompt)
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
