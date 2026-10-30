# Thin entry points; CMake presets own compiler, dependency and build settings.
.DEFAULT_GOAL := build
PYTHON ?= python3
PRESET ?= udoncl
BUILD_DIR ?= build/$(PRESET)
EXAMPLE_OUTPUT ?= build/opencl-host
BOUNDARY_OUTPUT ?= build/c-boundary
NATIVE_OUTPUT ?= build/native-fragment
OPENCL_OUTPUT ?= $(EXAMPLE_OUTPUT)

.PHONY: configure build test example example-c-boundary example-native-fragment example-opencl
configure:
	cmake --preset $(PRESET)

build: configure
	$(PYTHON) scripts/build.py --preset "$(PRESET)"

test: build
	ctest --preset $(PRESET)

example: example-opencl

example-c-boundary: build
	$(PYTHON) examples/c-boundary/run.py --build-dir "$(BUILD_DIR)" --output "$(BOUNDARY_OUTPUT)"

example-native-fragment: build
	$(PYTHON) examples/native-fragment/run.py --build-dir "$(BUILD_DIR)" --output "$(NATIVE_OUTPUT)"

example-opencl: build
	$(PYTHON) examples/opencl-host/run.py --build-dir "$(BUILD_DIR)" --output "$(OPENCL_OUTPUT)"
