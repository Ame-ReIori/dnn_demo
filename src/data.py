import csv
from array import array

def read_data(filename):
    csvFile = open(filename, 'r')
    reader = csv.reader(csvFile)

    train_data = []
    for item in reader:
        if reader.line_num == 1:
            continue
        for feature in item:
            train_data.append(float(feature))
    return train_data

origin = read_data('../data/feature.csv')
data = array('d', (t for t in origin))
print(data[0])
fp = open('../data/features.bin', 'wb')
data.tofile(fp)
fp.close()

origin = read_data('../data/label.csv')
data = array('d', (t for t in origin))
print(data[0])
fp = open('../data/label.bin', 'wb')
data.tofile(fp)
fp.close()