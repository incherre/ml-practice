'''A tool to provide human classifications of sentences.'''
import os
import csv

# Setting up the unrated file reader.
frompath = os.path.abspath(os.path.join('.', 'data', 'unrated_data.csv'))
fromfile = open(frompath, 'r', newline='', encoding='utf-8')
fromcsv = csv.reader(fromfile)

# Setting up the rated file writer.
topath = os.path.abspath(os.path.join('.', 'data', 'rated_data.csv'))
tofile = open(topath, 'a', newline='', encoding='utf-8')
tocsv = csv.writer(tofile)

# Setting up the skipped file writer.
skippath = os.path.abspath(os.path.join('.', 'data', 'skipped_data.csv'))
skipfile = open(skippath, 'w', newline='', encoding='utf-8')
skipcsv = csv.writer(skipfile)

skipping = False
print('y for yes, s for skip, blank for no, x for exit')
for row in fromcsv:
    if skipping:
        skipcsv.writerow(row)
        continue

    message = row[0]
    print('"' + message + '"')
    c = input()
    if c[:1].lower() == 'y':
        tocsv.writerow(row + ['1'])
    elif c[:1].lower() == 's':
        skipcsv.writerow(row)
    elif c[:1].lower() == 'x':
        skipcsv.writerow(row)
        skipping = True
    else:
        tocsv.writerow(row + ['0'])

fromfile.close()
tofile.close()
skipfile.close()
