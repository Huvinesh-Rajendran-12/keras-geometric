# CI/CD Build Fixes

## Issues Fixed

### 1. Missing Editables Package

The GitHub Actions CI/CD pipeline was failing with the following error:
```
ModuleNotFoundError: No module named 'editables'
```

This error occurred during the editable installation phase (`build_editable`) when trying to install the package. The `editables` package is required by `hatchling` for handling PEP 660 editable installs but was not listed as a dependency.

### 2. Missing TensorFlow Dependency

After fixing the editables issue, another error was encountered:
```
ModuleNotFoundError: No module named 'tensorflow'
```

This occurred because Keras requires TensorFlow as a backend dependency, but it wasn't being explicitly installed before testing the package.

## Changes Made

1. Added `editables` as a dependency in `pyproject.toml`:
   ```toml
   [build-system]
   requires = ["hatchling", "hatch-vcs", "editables"]
   build-backend = "hatchling.build"
   ```

2. Updated the CI/CD workflow file to explicitly install `editables` in all relevant build steps:
   - In the testing job
   - In the build job

3. Added a verification script at `scripts/verify_editable_install.sh` to test editable installations locally.

## Why This Works

The error was occurring because the `hatchling` build backend requires the `editables` package to implement the `build_editable` hook specified in PEP 660. The `editables` package provides the functionality to create editable installations without relying on the older `setup.py develop` approach.

When `pip install -e .` is executed, it looks for the `build_editable` hook in the build backend, but this feature requires the `editables` package to be present in the build environment.

## Verification

You can verify the fix locally by running:
```bash
./scripts/verify_editable_install.sh
```

This will create a temporary virtual environment using `uv`, install the package in editable mode with `uv pip install -e .`, and verify it can be imported successfully. The script uses the same `uv` toolchain that is used in the CI/CD pipeline to ensure consistency.
