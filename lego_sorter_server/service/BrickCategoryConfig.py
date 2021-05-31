import json


class BrickCategoryConfig:
    @staticmethod
    def conf_from_json(json_config):
        brick_cat_mapping = {}
        cat_positions = {}
        for cat in json_config:
            if cat in cat_positions:
                raise RuntimeError("Duplication of category unique name in BrickCategoryConfig .json file")
            if cat.lower() == "default":
                cat_positions["default"] = json_config[cat]["position"]
            else:
                cat_positions[cat] = json_config[cat]["position"]
                new_bricks = {brick: cat for brick in json_config[cat]["bricks"]}
                if len(new_bricks.keys() & brick_cat_mapping.keys()) != 0:
                    raise RuntimeError(F"Following bricks are redefined in category {cat}: {new_bricks.keys() & brick_cat_mapping.keys()}")
                brick_cat_mapping = {**brick_cat_mapping, **new_bricks}
        return brick_cat_mapping, cat_positions

    def __init__(self, config_path=None):
        if not config_path:
            self.brick_cat_mapping, self.cat_positions = {}, {}
        else:
            with open(config_path, "r") as src:
                self.brick_cat_mapping, self.cat_positions = BrickCategoryConfig.conf_from_json(json.load(src))

    def __getitem__(self, brick):
        if brick not in self.brick_cat_mapping:
            return self.__missing__(brick)
        else:
            cat = self.brick_cat_mapping[brick]
            return cat, self.cat_positions[cat]

    def __missing__(self, brick):
        return "default", self.cat_positions["default"]
