import os
import os.path as osp
import json
import codecs
import cPickle


def mkdir_if_missing(d):
    if not osp.isdir(d):
        os.makedirs(d)


def pickle(data, file_path):
    with open(file_path, 'wb') as f:
        cPickle.dump(data, f, cPickle.HIGHEST_PROTOCOL)


def unpickle(file_path):
    with open(file_path, 'rb') as f:
        data = cPickle.load(f)
    return data


def read_list(file_path, coding=None):
    if coding is None:
        with open(file_path, 'r') as f:
            arr = [line.strip() for line in f.readlines()]
    else:
        with codecs.open(file_path, 'r', coding) as f:
            arr = [line.strip() for line in f.readlines()]
    return arr


def write_list(arr, file_path, coding=None):
    if coding is None:
        arr = ['{}'.format(item) for item in arr]
        with open(file_path, 'w') as f:
            f.write('\n'.join(arr))
    else:
        with codecs.open(file_path, 'w', coding) as f:
            f.write(u'\n'.join(arr))


def read_kv(file_path, coding=None):
    arr = read_list(file_path, coding)
    if len(arr) == 0:
        return [], []
    return zip(*map(str.split, arr))


def write_kv(k, v, file_path, coding=None):
    arr = zip(k, v)
    arr = [' '.join(item) for item in arr]
    write_list(arr, file_path, coding)


def read_json(file_path):
    with open(file_path, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, file_path):
    with open(file_path, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))