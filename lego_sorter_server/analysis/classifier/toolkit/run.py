import argparse
import json
import random
import os
import shutil
from pydoc import locate
from datetime import datetime


class DataAlchemist():
    def __init__(self, config_path):
        with open(config_path, "r") as src:
            self.cfg = json.load(src)
        self.source_path = None
        self.classes = None
        self.processor = None
        self.types = None
        self.transformations = None
        self.dst = None
        self.div_unit = None
        self.settings_from_config()

    def settings_from_config(self):
        random.seed(self.get_cfg("RANDOM_SEED") or datetime.now())
        self.source_path = self.get_cfg("SOURCE_PATH") or os.path.join("images", "storage", "stored")
        self.classes = self.get_cfg("INCLUDED_CLASSESS") or os.listdir(self.source_path)
        if "original" in self.classes:
            self.classes.remove("original")
        if self.get_cfg("EXCLUDED_CLASSESS"):
            for cls in self.get_cfg("EXCLUDED_CLASSESS"):
                self.classes.remove(cls)
        self.classes = list(self.classes)
        if self.get_cfg("RANDOM_CLASSESS_COUNT"):
            random.shuffle(self.classes)
            self.classes = self.classes[:self.get_cfg("RANDOM_CLASSESS_COUNT")]
        transformations = self.get_cfg("TRANSFORMATIONS") or []
        self.transformations = [locate(f"transformations.{transformation.lower()}.{transformation}") for transformation in transformations]
        processor = self.get_cfg("PROCESSOR") or "Captured"
        self.processor = locate(f"processors.{processor.lower()}.{processor}")
        self.types = self.get_cfg("TYPES")
        if not self.types:
            raise RuntimeError("key: `TYPES` is required. Please check README")
        self.div_unit = self.get_cfg("DIV_UNIT") or "%"
        self.dst = self.get_cfg("DESTINATION") or "DataSet"
        if self.get_cfg("CLEAR_DESTINATION_BEFORE") and os.path.isdir(self.dst):
            shutil.rmtree(self.dst, ignore_errors=True)

    def get_cfg(self, key):
        return self.cfg[key] if key in self.cfg else None

    def perform(self):
        self.processor.precalc_sizes(self.source_path, self.classes, self.types, self.div_unit)
        limit = len(self.classes)
        i = 1
        for cls in self.classes:
            print(F"processed: {i} of {limit} classes ({cls})")
            self.processor.run(os.path.join(self.source_path, cls), self.dst, cls, self.types,
                               self.transformations)
            i += 1

    def print_classes(self):
        print("Selected classes:")
        print((", ").join(self.classes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process dataset with DataAlchemist')
    parser.add_argument('config_path', type=str,
                        help='path to config .json file')

    args = parser.parse_args()
    dataAlchemist = DataAlchemist(args.config_path)
    dataAlchemist.perform()
