# Thin entry points; CMake presets own compiler, dependency and build settings.
.DEFAULT_GOAL := build
PYTHON ?= python3
PRESET ?= udoncl
BUILD_DIR ?= build/$(PRESET)
EXAMPLE_OUTPUT ?= build/c-boundary

.PHONY: configure build test example example-c-boundary
configure:
	cmake --preset $(PRESET)

build: configure
	cmake --build --preset $(PRESET)

test: build
	ctest --preset $(PRESET)

example: build
	$(PYTHON) examples/c-boundary/run.py --build-dir "$(BUILD_DIR)" --output "$(EXAMPLE_OUTPUT)"

example-c-boundary: example
