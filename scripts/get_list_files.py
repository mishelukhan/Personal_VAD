import os
import sys

def get_list_files(path: str, mod='train'):
    file_list = []
    for dirpath, dirnames, filenames in os.walk(path):
        file_list = dirnames
        break
    name_file = mod + '_list.txt'
    with open(name_file, 'w') as f:
        for each in file_list:
            f.write(each + '\n')


if __name__ == '__main__':
    path = '/mnt/c/projects/Personal_VAD/amicorpus'
    mod = 'train'
    get_list_files(path, mod)
