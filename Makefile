# Cuda.JS Makefile

.PHONY: build clean test install dev examples

# Default target
all: build

# Install dependencies
install:
	npm install

# Build the project
build: install
	npm run build

# Clean build artifacts
clean:
	npm run clean
	rm -rf node_modules coverage

# Run tests
test: build
	npm test

# Development build with watch
dev: install
	npm run build:ts -- --watch

# Run examples
examples: build
	@echo "Running basic..."
	npm run example:basic

# Check CUDA installation
check-cuda:
	@echo "Checking CUDA installation..."
	@command -v nvcc >/dev/null 2>&1 || { echo "nvcc not found. Please install CUDA Toolkit."; exit 1; }
	@nvcc --version
	@echo "CUDA installation OK"

# Setup development environment
setup: check-cuda install build
	@echo "CudaJS development environment ready!"

# Release build
release: clean build test
	npm pack

# Benchmark
benchmark: build
	@echo "Running benchmarks..."
	node examples/basic.js

# Help
help:
	@echo "Cuda.JS Build System"
	@echo ""
	@echo "Targets:"
	@echo "  install    - Install dependencies"
	@echo "  build      - Build the project"  
	@echo "  clean      - Clean build artifacts"
	@echo "  test       - Run test suite"
	@echo "  dev        - Development build with watch"
	@echo "  examples   - Run example programs"
	@echo "  check-cuda - Verify CUDA installation"
	@echo "  setup      - Complete development setup"
	@echo "  release    - Build release package"
	@echo "  benchmark  - Run performance benchmarks"
	@echo "  help       - Show this help"