import os
import random
import shutil

from PIL import Image
from numpy.random import choice
from shutil import copyfile

from lego_sorter_server.analysis.classifier.toolkit.processors.processor import Processor
from lego_sorter_server.analysis.classifier.toolkit.transformations.transformation import TransformationException


class Renders(Processor):
    @staticmethod
    def get_series_id(name):
        return name.rsplit('_', 2)[0]

    @staticmethod
    def precalc_sizes(src, classes, types, div_unit):
        min_size = min([len(os.listdir(os.path.join(src, cls))) for cls in classes])
        print(F"minimal_size: {min_size}")
        if div_unit == "%":
            for type, content in types.items():
                if "DIV_VALUE" in content and content["DIV_VALUE"]:
                    content["_val"] = int(content["DIV_VALUE"] * min_size)
                else:
                    content["_val"] = None
        else:
            raise NotImplementedError(F"DIV_UNIT: {div_unit} is not supported by Renders Processor")

    @staticmethod
    def calc_probs(candidates):
        cands = [x for x in candidates]
        missing = []
        index_of_rest = None
        for cand in candidates:
            if candidates[cand]["max"]:
                missing.append(candidates[cand]["max"] - candidates[cand]["curr"])
            else:
                index_of_rest = cands.index(cand)
                missing.append(0)

        if not any(missing):
            probs = missing
            if not index_of_rest:
                return False, None
            probs[index_of_rest] = 1
        else:
            probs = []
            all_missing = sum(missing)
            for single in missing:
                probs.append(single / all_missing)

        return True, probs

    @staticmethod
    def run(src, dst, cls, types, transformations):
        print(F"class: {src}")
        for type in types:
            dst_dir = os.path.join(dst, type, cls)
            if not os.path.isdir(dst_dir):
                os.makedirs(dst_dir)

        files = os.listdir(src)
        series_target = {}
        # random.shuffle(files)
        candidates = {
            tp: {"curr": 0, "max": content["_val"]} for tp, content in types.items()
        }
        for file in files:
            series_id = Renders.get_series_id(file)
            if series_id in series_target:
                curr_candidates = {series_target[series_id]: candidates[series_target[series_id]]}
            else:
                curr_candidates = candidates
            status, probs = Renders.calc_probs(curr_candidates)
            if status == False:
                continue
            cand_names = [name for name in curr_candidates]
            target = choice(cand_names, 1, p=probs)[0]
            try:
                src_file = os.path.join(src, file)
                dst_dir = os.path.join(dst, target, cls)
                dst_file = os.path.join(dst_dir, file)
                if not os.path.isdir(dst_dir):
                    os.makedirs(dst_dir)
                im = Image.open(src_file)
                for transformation in transformations:
                    im = transformation.transform(im)
                im.save(dst_file)
                series_target[series_id] = target
                candidates[target]["curr"] += 1
            except TransformationException as ex:
                print(F"TransformationException: {file}. Skipping")
                print(ex)
                src_file = os.path.join(src, file)
                wrong_dir = os.path.join(dst, "wrong")
                if not os.path.isdir(wrong_dir):
                    os.makedirs(wrong_dir)
                shutil.copyfile(src_file, os.path.join(wrong_dir, ex.prefix+"_"+file))
            except Exception as ex:
                print(F"Unable to transform file: {file}. Skipping")
                print(ex)