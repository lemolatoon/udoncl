import subprocess
import sys
from pathlib import Path

# The elements of a cl_mem array are tracked cells of their own once the
# loops over the array are unrolled.
optimizer = Path(sys.argv[1])
source = Path(sys.argv[2])
command = [
    optimizer,
    "--vrc-unroll-static-resource-loops",
    "--vrc-discover-static-device-resources",
    "--vrc-propagate-device-resource-provenance",
    source,
]
result = subprocess.run(command, text=True, capture_output=True)
assert result.returncode == 0, (result.stdout, result.stderr)
lines = result.stdout.splitlines()
summary = next(line for line in lines
               if "vrc.static_device_global_cell_provenance" in line)
# One cell per element, with its byte offset: the allocation of the loop's
# iteration, or of the helper given the element's address.
for offset, resource in [(0, 0), (4, 1), (8, 2)]:
    cell = (f"{{global = @buffers, may_be_null = true, offset = {offset} : "
            f"i32, resource_ids = [{resource} : i32], unknown = false}}")
    assert cell in summary, summary
# A constant element reads the buffer of that element.
read = next(line for line in lines if "call @read" in line)
assert "resource_ids = [1 : i32]" in read, read
assert "unknown = false" in read, read
# The release loop releases each element's own buffer.
releases = [line for line in lines if "call @clReleaseMemObject" in line]
assert len(releases) == 3, releases
for resource, line in enumerate(releases):
    assert f"resource_ids = [{resource} : i32]" in line, line
print("static_cl_mem_array=passed elements=3")
