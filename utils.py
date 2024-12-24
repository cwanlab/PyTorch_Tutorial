# data: http://download.pytorch.org/tutorial/data.zip

import io
import os
import unicodedata
import string
import glob

import torch
import random

ALL_LETTERS = string.ascii_letters + " .,;'" # all possible letters
N_LETTERS = len(ALL_LETTERS)


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != "Mn"
        and c in ALL_LETTERS
    )

"""
To represent a single letter, we use a “one-hot vector” of 
size <1 x n_letters>. A one-hot vector is filled with 0s
except for a 1 at index of the current letter, e.g. "b" = <0 1 0 0 0 ...>.

To make a word we join a bunch of those into a
2D matrix <line_length x 1 x n_letters>.

That extra 1 dimension is because PyTorch assumes
everything is in batches - we’re just using a batch size of 1 here.
"""

# Find letter index from all_letters, e.g. "a" = 0
def letter_to_index(letter):
    return ALL_LETTERS.find(letter)

def letter_to_tensor(letter):
    tensor = torch.zeros(1, N_LETTERS)
    tensor[0][letter_to_index(letter)] = 1
    return tensor
    
# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, N_LETTERS)
    for i, letter in enumerate(line):
        tensor[i][0][letter_to_index(letter)] = 1
    return tensor

def load_data():
    category_lines = {} # Build the category_lines dictionary {country: a list of names}
    all_categories = [] # A list of countries
    
    def find_files(path):
        return glob.glob(path)
    
    # Read a file and split into lines
    def read_lines(filename):
        lines = io.open(filename, encoding='utf-8').read().strip().split('\n')
        return [unicode_to_ascii(line) for line in lines]
    
    for filename in find_files('data/data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        
        lines = read_lines(filename)
        category_lines[category] = lines
        
    return category_lines, all_categories

def random_training_example(category_lines, all_categories):
    
    # choose one
    def random_choice(a):
        random_idx = random.randint(0, len(a) - 1)
        return a[random_idx]
    
    # randomly choose one category
    category = random_choice(all_categories)
    # randomly choose one name
    line = random_choice(category_lines[category])

    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    # print(category, category_tensor)
    
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor


if __name__ == '__main__':
    print(ALL_LETTERS)
    print(N_LETTERS)
    print(unicode_to_ascii('Ślusàrski'))

    print("the index of a:", letter_to_index("a"))
    print(line_to_tensor("acd")[:5, :, :5])

    category_lines, all_categories = load_data()
    print(all_categories)
    print("Chinese:", category_lines["Chinese"][:5])

    random_training_example(category_lines, all_categories)
