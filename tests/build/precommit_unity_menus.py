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
for menu in standard:
    assert f'[MenuItem("{menu}")]' in source, menu
# Package Samples compile against the bindings of the imported examples.
samples = "Tools/udoncl/Internal/Prepare Package Sample Validation"
assert standard.index(samples) > standard.index(
    "Tools/udoncl/Internal/Prepare Packaged Example Validation"
), standard
# The gate checks that the imported examples installed the bindings the
# SAXPY and MNIST Package Samples use.
sample_sources = [
    path.read_text(encoding="utf-8")
    for path in (ROOT / "unity/Packages/com.lemolatoon.udoncl/Samples~").rglob("*.cs")
]
for path in precommit.PACKAGED_EXAMPLE_BINDINGS:
    binding = Path(path)
    assert binding.parent.name == "Bindings", path
    assert any(
        f"private {binding.stem} binding;" in source for source in sample_sources
    ), path
print("precommit_unity_menus=passed")
