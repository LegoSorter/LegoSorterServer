import random
import shutil
from math import floor

DIR = R"F:\slawek\final_dataset_prepared_447_classes\final_dataset\val"
OUT_1 = R"F:\slawek\enh\real_val"
OUT_2 = R"F:\slawek\enh\real_train"

import os

clses = os.listdir(DIR)
clses = [x for x in clses if '.' not in x]

mx = 0
for cls in clses:
    files = os.listdir(os.path.join(DIR, cls))
    count = len(files)
    if mx < count:
        mx = count


for cls in clses:
    os.mkdir(os.path.join(OUT_1, cls))
    os.mkdir(os.path.join(OUT_2, cls))
    files = os.listdir(os.path.join(DIR, cls))
    res = files
    res = random.sample(res, floor(len(res)/2))
    res = set(res)
    for file in res:
        dst = os.path.join(OUT_1, cls, file)
        shutil.copy(os.path.join(DIR, cls, file), dst)

    res = set(files)-res
    print(res)
    for file in res:
        dst = os.path.join(OUT_2, cls, file)
        shutil.copy(os.path.join(DIR, cls, file), dst)

