'''A tool to mine difficult examples from the pile.eleuther.ai dataset.'''
import classify
import os
import csv
import json
import nltk

# Options.
difficulty_increment = 0.0001
difficulty_multiple = 363  # Experimentally determined.
determine_difficulty_multiple = False

number_of_lines = 7021438
starting_index = 0
ending_index = 70222  # ~1%.

# Download nltk sentence tokenizer.
nltk.download('punkt')

# Set up the file writer.
topath = os.path.abspath(os.path.join('.', 'data', 'unrated_data.csv'))
tofile = open(topath, 'w', newline='', encoding='utf-8')
tocsv = csv.writer(tofile)

# Set up the input file.
frompath = os.path.abspath('D:\\00.jsonl')
fromfile = open(frompath, 'r')

# Map to remove newlines.
punct_map = {}
punct_map[ord('\n')] = ' '


line_num = 0
try:
    if determine_difficulty_multiple:
        difficulty_multiple = 0

    for count, line in enumerate(fromfile):
        line_num = count
        if count < starting_index and not determine_difficulty_multiple:
            continue
        if count >= ending_index and not determine_difficulty_multiple:
            break

        if count % (number_of_lines // 100000) == 0:
            print('Percent complete: {:.3%}, Current line: {}, Flushing file...'.format(count / number_of_lines, count))
            tofile.flush()

        if determine_difficulty_multiple and count >= 100:
            break

        json_obj = json.loads(line)
        if not 'text' in json_obj:
            print('No text.')
            continue

        for text in nltk.tokenize.sent_tokenize(json_obj['text']):
            text = text.translate(punct_map).strip()
            if text == '':
                continue

            chance_positive = classify.classify_sentence(text)
            current_tolerance = difficulty_multiple * difficulty_increment
            if chance_positive < (0.5 - current_tolerance) or chance_positive > (0.5 + current_tolerance):
                if determine_difficulty_multiple:
                    difficulty_multiple += 1
                continue

            if determine_difficulty_multiple:
                print('Current difficulty multiple: ', difficulty_multiple)
                difficulty_multiple //= 2
            tocsv.writerow([text, str(chance_positive)])
except KeyboardInterrupt:
    print('Interrupted during line {}.'.format(line_num))

fromfile.close()
tofile.close()
input('Press enter to exit.')
