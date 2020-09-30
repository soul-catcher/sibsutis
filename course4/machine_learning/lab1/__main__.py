import csv
import math
import random
import typing
from collections import defaultdict


def split_by_classes(raw_data: typing.Iterable[typing.Sequence[int]]) -> typing.DefaultDict[int, typing.List[int]]:
    """Разделяет данные на классы, к которым они относятся"""
    splitted_data = defaultdict(list)
    for item in raw_data:
        splitted_data[item[-1]].append(item)
    return splitted_data


def split_by_training_and_test_sets(
    raw_data: typing.Iterable[typing.Sequence[int]],
) -> typing.Tuple[typing.List[typing.List[int]], typing.List[typing.List[int]]]:
    """Разделяет входной набор данные на обучающий и тестовый набор"""
    splitted_data = split_by_classes(raw_data)
    training_set, test_set = [], []
    for data_class, items in splitted_data.items():
        random.shuffle(items)
        size_of_test_set = len(items) // 3
        test_set += items[:size_of_test_set]
        training_set += items[size_of_test_set:]
    return training_set, test_set


def quart_core(r: float) -> float:
    """Квартическое ядро"""
    return (1 - r ** 2) ** 2 if abs(r) < 1 else 0


def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Возвращает расстояние между двумя точками на плоскости"""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def expected_value(data: typing.Collection[float]) -> float:
    """Возвращает математическое ожидание случайной величины"""
    return sum(data) / len(data)


def std_deviation(data: typing.Collection[float], expected: float):
    """Возвращает среднекрвадратическое отклонение"""
    return math.sqrt(sum((elem - expected) ** 2 for elem in data) / len(data))


def normalize_data(data: typing.Collection[float], expected: float, deviation: float) -> typing.List[float]:
    """Z-score нормализация данных"""
    return [(x - expected) / deviation for x in data]


def remove_doubles(data):
    return list(set(data))


raw_data = csv.reader(open("data5.csv"))
next(raw_data)
raw_data = list(raw_data)
random.shuffle(raw_data)
raw_data = raw_data[:500]  # Берём лишь кусок исходного набора для ускорения работы
converted_data = (tuple(map(int, line)) for line in raw_data)
train_set, t_set = split_by_training_and_test_sets(converted_data)

train_set = remove_doubles(train_set)

mrot_in_hour_list = [x[0] for x in train_set]
salary_list = [x[1] for x in train_set]

mrot_in_hour_expected, salary_expected = expected_value(mrot_in_hour_list), expected_value(salary_list)
mrot_in_hour_deviation, salary_deviation = (
    std_deviation(mrot_in_hour_list, mrot_in_hour_expected),
    std_deviation(salary_list, salary_expected),
)

mrot_in_hour_normalized = normalize_data(mrot_in_hour_list, mrot_in_hour_expected, mrot_in_hour_deviation)
salary_normalized = normalize_data(salary_list, salary_expected, salary_deviation)

normalized_train_set = [
    [mrot_in_hour_normalized[i], salary_normalized[i], train_set[i][-1]] for i in range(len(train_set))
]


def calc_distances(dot, all_set):
    distances = []
    for d in all_set:
        distances.append(distance(dot[0], dot[1], d[0], d[1]))
    return distances


def calc_hits(k: int, set):
    hits = 0
    for dot in set:
        distances = calc_distances(dot, set)
        z = list(zip(distances, set))
        z.sort(key=lambda x: x[0])
        assert z[0][0] == 0.0
        neighbours = z[1 : k + 2]
        max_range = neighbours[-1][0]
        s = 0
        for neigh in neighbours:
            if neigh[-1][-1] == 1:
                mult = 1
            elif neigh[-1][-1] == 0:
                mult = -1
            qc = quart_core(neigh[0] / max_range) * mult
            s += qc
        if s < 0 and dot[-1] == 0 or s > 0 and dot[-1] == 1:
            hits += 1
    return hits


max_val = 0
max_k = 0
for k in range(1, len(normalized_train_set) - 1):
    a = calc_hits(k, normalized_train_set)
    print(f"k = {k}, hits = {a} / {len(normalized_train_set)}")
    if a > max_val:
        max_val = a
        max_k = k

print("max k", max_k)

test_mrot_in_hour_list = [x[0] for x in t_set]
test_salary_list = [x[1] for x in t_set]
test_mrot_in_hour_normalized = normalize_data(test_mrot_in_hour_list, mrot_in_hour_expected, mrot_in_hour_deviation)
test_salary_normalized = normalize_data(test_salary_list, salary_expected, salary_deviation)


def get_class_of_dot(k, set, dot):
    distances = calc_distances(dot, set)
    z = list(zip(distances, set))
    z.sort(key=lambda x: x[0])
    neighbours = z[: k + 2]
    max_range = neighbours[-1][0]
    s = 0
    for neigh in neighbours:
        if neigh[-1][-1] == 1:
            mult = 1
        elif neigh[-1][-1] == 0:
            mult = -1
        qc = quart_core(neigh[0] / max_range) * mult
        s += qc
    if s < 0:
        return -1
    else:
        return 1


normalized_test_set = [
    [test_mrot_in_hour_normalized[i], test_salary_normalized[i], t_set[i][-1]] for i in range(len(t_set))
]
hits = 0
for dot in normalized_test_set:
    dot_class = -1 if dot[-1] == 0 else 1
    if get_class_of_dot(max_k, normalized_train_set, dot) == dot_class:
        hits += 1
print(f"Test set accuracy: {hits}/{len(normalized_test_set)} - {hits / len(normalized_test_set) * 100:.3}%")
