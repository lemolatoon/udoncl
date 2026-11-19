from pathlib import Path
import sys


ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT / "scripts"))

import precommit  # noqa: E402


scene = precommit.ENSURE_VALIDATION_SCENE_MENU
standard = precommit.final_preparation_menus(True)
focused = precommit.final_preparation_menus(False)

# The batchmode test runner needs a saved scene in every flow. Focused flows
# skip the standard fixtures but must still save the validation scene last.
assert standard[-1] == scene, standard
assert standard[:-1] == list(precommit.STANDARD_FIXTURE_MENUS), standard
assert focused == [scene], focused
assert standard.count(scene) == 1, standard

source = (ROOT / "unity/Packages/com.lemolatoon.udoncl/Editor/"
          "UdonclPackagedExampleImporter.cs").read_text(encoding="utf-8")
assert f'[MenuItem("{scene}")]' in source
print("precommit_unity_menus=passed")
