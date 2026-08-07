from pathlib import Path

# import robosuite_model_zoo
import os
import shutil


def get_ext(path):
    return Path(path).suffix


def get_stem(path):
    return Path(path).stem


def get_name(path):
    return Path(path).name


def make_asset_path(base_asset_path, model_prefix, model_name):
    if base_asset_path is None:
        base_asset_path = os.path.join(
            os.path.dirname(robosuite_model_zoo.__file__),
            "assets_private",
        )
    if model_prefix is not None:
        base_asset_path = os.path.join(base_asset_path, model_prefix)
    asset_path = os.path.join(base_asset_path, model_name)

    if os.path.exists(asset_path):
        shutil.rmtree(asset_path)

    os.makedirs(asset_path)

    return asset_path
