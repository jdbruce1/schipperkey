cmaj = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19 ,2.39, 3.66, 2.29, 2.88)
cmin = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)
offset_map = {"C": 0, "Db": 1, "D": 2, "Eb": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "Ab": 8, "A": 9, "Bb": 10, "B": 11}


def get_key_vector(note, mode):
    if mode == "major":
        return roll(cmaj, offset_map[note])
    elif mode == "minor":
        return roll(cmin, offset_map[note])
    return None


def roll(vector, offset):
    if offset >= len(vector):
        raise ValueError("Offset must be less than vector length.")
    return vector[offset:] + vector[:offset]
