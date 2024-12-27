import os
import re

_filename_ascii_strip_re = re.compile(r"[^A-Za-z0-9_.,=-]")


# Adapted from https://tedboy.github.io/flask/_modules/werkzeug/utils.html#secure_filename
# Changed in order to work without importing flask
def sanitize_shorten_ckppath(path):
    path = path.replace("hyperparams.", "")
    path = path.replace("training.", "")
    path = path.replace("+experiments", "exps")
    path = path.replace("num_epochs", "eps")
    path = path.replace("batch_size", "bs")
    path = path.replace("forecast_steps", "frc_steps")
    path = path.replace("loss_params.", "")
    path = path.replace("action", "actn")
    path = path.replace("forecast", "frc")
    path = path.replace("classification", "cls")
    path = path.replace("state", "stt")
    path = path.replace("dataset", "dts")
    path = path.replace("subsample_ratio", "smplrtio")
    path = path.replace("reconstruction", "rcns")

    for sep in os.path.sep, os.path.altsep:
        if sep:
            path = path.replace(sep, " ")
    path = str(_filename_ascii_strip_re.sub("", "_".join(path.split()))).strip("._")
    return path


def config_init():
    from dataclasses import dataclass

    from omegaconf import MISSING, OmegaConf

    import hydra
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()

    def merge_keys(*cfg):
        all_keys = set()
        for c in cfg:
            all_keys.update([str(x) for x in c.keys()])
        return "-".join(sorted(all_keys))

    # If has key, return True, else return False
    def has_key(cfg, key):
        return key in cfg

    def get_key(cfg, key, *args):
        # If args is not empty, recurse after getting the key
        if len(args) > 0:
            if key in cfg:
                try:
                    return get_key(cfg[key], *args)
                except KeyError:
                    raise KeyError(f"Key {args} not found in {cfg[key]}")
            else:
                raise KeyError(f"Key {key} not found in {cfg}")

        if key in cfg:
            return cfg[key]
        else:
            raise KeyError(f"Key {key} not found in {cfg}")

    def resolve_quantizer_path(keys, plant_data, quantizer_dict):
        quantizer_key = "plant" if plant_data else "legacy"
        return get_key(quantizer_dict, quantizer_key, keys)

    OmegaConf.register_new_resolver("merge_keys", merge_keys)

    OmegaConf.register_new_resolver("has_key", has_key)

    OmegaConf.register_new_resolver("get_key", get_key)
    OmegaConf.register_new_resolver("eval", eval)

    bool_to_str = lambda x: "true" if x else "false"
    OmegaConf.register_new_resolver("bool_to_str", bool_to_str)
    OmegaConf.register_new_resolver("resolve_quantizer_path", resolve_quantizer_path)
    OmegaConf.register_new_resolver("sanitize", sanitize_shorten_ckppath)
