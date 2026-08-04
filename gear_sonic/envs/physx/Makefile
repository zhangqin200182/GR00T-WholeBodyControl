# Convenience: build PhysX pybind11 module on aarch64.
# Prerequisites: cmake, pybind11, PhysX 5 SDK at /opt/physx/physx

.PHONY: build clean test

BUILD_DIR := build

build:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$$(nproc)
	@echo "=== Build complete ==="
	@echo "Try:  python -c 'import physx_core; print(\"OK\")'"

clean:
	rm -rf $(BUILD_DIR)

test:
	python -c "import sys; sys.path.insert(0, '$(CURDIR)/$(BUILD_DIR)'); import physx_core; print('OK')"
