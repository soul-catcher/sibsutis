import csv
import typing
import random
from collections import defaultdict


def split_by_classes(raw_data: typing.Iterable[typing.Sequence]) -> typing.DefaultDict[typing.Hashable, list]:
    splitted_data = defaultdict(list)
    for item in raw_data:
        splitted_data[item[-1]].append(item[:-1])
    return splitted_data


def split_by_training_and_test_sets(raw_data: typing.Iterable[typing.Sequence]
) -> typing.Tuple[typing.Dict[typing.Hashable, list], ...]:
    splitted_data = split_by_classes(raw_data)
    training_set, test_set = {}, {}
    for data_class, items in splitted_data.items():
        random.shuffle(items)
        size_of_test_set = len(items) // 3
        test_set[data_class] = items[:size_of_test_set]
        training_set[data_class] = items[size_of_test_set:]
    return training_set, test_set


data = list(csv.reader(open("data5.csv")))
next(data)
training_set, test_set = split_by_training_and_test_sets(data)
