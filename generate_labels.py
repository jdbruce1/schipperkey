import os
import csv

out_file = open("dataset_metadata.csv", 'wb')
writer = csv.writer(out_file, delimiter=',')
for f in os.listdir("full_dataset"):
    lst = []
    name = os.path.basename(f)
    name = os.path.splitext(name)[0].strip()
    lst.append(name)
    splt = name.split("_")
    splt[-1] = splt[-1][:1].upper() + splt[-1][1:]
    if "pirates" in name or "potter" in name or "sins" in name:
        lst.append(splt[-1] + 'm')
    else:
        lst.append(splt[-1])
    lst.append("Will")
    if len(splt) > 2:
        lst.append(splt[-2])
    else:
        lst.append("h")
    writer.writerow(lst)

out_file.close()
