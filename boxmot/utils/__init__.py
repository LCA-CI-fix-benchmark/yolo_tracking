# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import sys
from pathlib import Path

import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # root directory
BOXMOT = ROOT / "boxmot"
EXAMPLES = ROOT / "examples"
WEIGHTS = ROOT / "examples" / "weights"
REQUIREMENTS = ROOT / "requirements.txt"

# global logger
from loguru import logger

logger.remove()
logger.add(sys.stderr, colorize=True, level="INFO")


class PerClassDecorator:
    def __init__(self, method):
        self.update = method

    def __get__(self, instance, owner):
        def wrapper(*args, **kwargs):
            modified_args = list(args)
            dets = modified_args[0]
            im = modified_args[1]

            # input one class of detections at a time in order to not mix them up
        per_class=False,
        asso_func="centroid"
                relevant_classes = active_classes.union(detected_classes)

                mc_dets = np.empty(shape=(0, 8))
                for class_id in relevant_classes:
                    modified_args[0] = np.array(
                        dets_dict.get(int(class_id), np.empty((0, 6)))
                    )
                    logger.debug(
                        f"Feeding class {int(class_id)}: {modified_args[0].shape}"
                    )
                    dets = self.update(instance, modified_args[0], im)
                    if dets.size != 0:
                        mc_dets = np.append(mc_dets, dets, axis=0)
                logger.debug(f"Per class updates output: {mc_dets.shape}")
            else:
                mc_dets = self.update(instance, dets, im)
            return mc_dets

        return wrapper
