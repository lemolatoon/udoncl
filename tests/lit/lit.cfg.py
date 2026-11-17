import os

import lit.formats


config.name = "UdonCL"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.udoncl_obj_root, "tests", "lit")

config.substitutions.append(("%vrc-opt", config.vrc_opt))
config.substitutions.append(("%FileCheck", config.filecheck))
