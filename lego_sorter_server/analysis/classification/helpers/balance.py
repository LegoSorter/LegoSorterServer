import random
import shutil

DIR = R"F:\slawek\enh\real_val"
OUT = R"F:\slawek\enh\real_val_balance"

import os

clses = os.listdir(DIR)
clses = [x for x in clses if '.' not in x]

mx = 0
for cls in clses:
    files = os.listdir(os.path.join(DIR, cls))
    count = len(files)
    if mx < count:
        mx = count

print(mx)
mx = min(100, mx)
for cls in clses:
    os.mkdir(os.path.join(OUT, cls))
    files = os.listdir(os.path.join(DIR, cls))
    res = files
    if len(res) < mx:
        while len(res) < mx:
            res += files
    res = random.sample(res, mx)
    id = 0
    for file in res:
        dst = os.path.join(OUT, cls, F"{id}_{file}")
        shutil.copy(os.path.join(DIR, cls, file), dst)
        id += 1

