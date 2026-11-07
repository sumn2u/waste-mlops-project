"""
Data validation for the garbage classification dataset.
Run BEFORE training to catch silent failures in labels/images.

Updated for Great Expectations 1.23.2 with idempotent datasource creation.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Use Great Expectations if available, otherwise basic checks
try:
    import great_expectations as gx
    HAS_GE = True
except ImportError:
    HAS_GE = False
    print("Great Expectations not installed. Using basic validation.")


def validate_yolo_labels(data_root: str) -> bool:
    """
    Validate YOLO format labels: each .txt file should contain lines
    with format: <class_id> <x_center> <y_center> <width> <height>
    All values normalized between 0 and 1.
    """
    labels_dir = Path(data_root) / "labels" / "train"
    if not labels_dir.exists():
        print(f"ERROR: Labels directory not found: {labels_dir}")
        return False

    label_files = list(labels_dir.glob("*.txt"))
    if len(label_files) == 0:
        print("ERROR: No label files found")
        return False

    print(f"Found {len(label_files)} label files")

    errors = []
    class_ids_seen = set()

    for label_file in label_files:
        with open(label_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) != 5:
                    errors.append(
                        f"{label_file.name}:{line_num} - "
                        f"expected 5 values, got {len(parts)}"
                    )
                    continue

                try:
                    class_id = int(parts[0])
                    coords = [float(x) for x in parts[1:5]]
                except ValueError:
                    errors.append(
                        f"{label_file.name}:{line_num} - non-numeric values"
                    )
                    continue

                # Class ID should be 0-6 (7 classes)
                if class_id < 0 or class_id > 6:
                    errors.append(
                        f"{label_file.name}:{line_num} - "
                        f"invalid class_id: {class_id}"
                    )

                # Coordinates should be normalized (0-1)
                for i, coord in enumerate(coords):
                    if coord < 0 or coord > 1:
                        errors.append(
                            f"{label_file.name}:{line_num} - "
                            f"coord[{i}] out of range: {coord}"
                        )

                class_ids_seen.add(class_id)

    if errors:
        print("VALIDATION FAILED:")
        for err in errors[:20]:  # Show first 20 errors
            print(f"  - {err}")
        return False

    print(f"Validation passed. Classes seen: {sorted(class_ids_seen)}")
    return True


def validate_data_with_great_expectations(data_root: str) -> bool:
    """
    Use Great Expectations 1.23.2 for more thorough validation.
    """
    if not HAS_GE:
        return validate_yolo_labels(data_root)

    # ============================================================
    # STEP 1: Load labels into a DataFrame
    # ============================================================
    labels_dir = Path(data_root) / "labels" / "train"
    rows = []

    for label_file in labels_dir.glob("*.txt"):
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    rows.append({
                        "class_id": int(parts[0]),
                        "x_center": float(parts[1]),
                        "y_center": float(parts[2]),
                        "width": float(parts[3]),
                        "height": float(parts[4]),
                    })

    if not rows:
        print("No valid label rows found")
        return False

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} label rows for validation")

    # ============================================================
    # STEP 2: Get a Great Expectations context
    # ============================================================
    print("Setting up Great Expectations 1.x validation...")
    context = gx.get_context(mode="file")

    # ============================================================
    # STEP 3: Create or Retrieve pandas Data Source (IDEMPOTENT)
    # ============================================================
    # This prevents the "datasource already exists" error on re-runs.
    datasource_name = "label_validation_source"

    try:
        # Try to retrieve an existing datasource first
        datasource = context.data_sources.get(datasource_name)
        print(f"Using existing datasource: {datasource_name}")
    except Exception:
        # Create it if it doesn't exist
        datasource = context.data_sources.add_pandas(name=datasource_name)
        print(f"Created new datasource: {datasource_name}")

    # ============================================================
    # STEP 4: Create or Retrieve Data Asset (IDEMPOTENT)
    # ============================================================
    asset_name = "label_asset"

    try:
        data_asset = datasource.get_asset(asset_name)
        print(f"Using existing data asset: {asset_name}")
    except Exception:
        data_asset = datasource.add_dataframe_asset(name=asset_name)
        print(f"Created new data asset: {asset_name}")

    # ============================================================
    # STEP 5: Create or Retrieve Batch Definition (IDEMPOTENT)
    # ============================================================
    batch_definition_name = "label_batch_definition"

    try:
        batch_definition = data_asset.get_batch_definition(batch_definition_name)
        print(f"Using existing batch definition: {batch_definition_name}")
    except Exception:
        batch_definition = data_asset.add_batch_definition_whole_dataframe(
            batch_definition_name
        )
        print(f"Created new batch definition: {batch_definition_name}")

    # ============================================================
    # STEP 6: Get the Batch (pass the DataFrame here)
    # ============================================================
    batch = batch_definition.get_batch(
        batch_parameters={"dataframe": df}
    )

    # ============================================================
    # STEP 7: Create or Retrieve Expectation Suite (IDEMPOTENT)
    # ============================================================
    suite_name = "yolo_label_suite"

    try:
        suite = context.suites.get(suite_name)
        print(f"Using existing expectation suite: {suite_name}")
    except Exception:
        suite = context.suites.add(
            gx.ExpectationSuite(name=suite_name)
        )
        print(f"Created new expectation suite: {suite_name}")

    # ============================================================
    # STEP 8: Get a Validator using the batch
    # ============================================================
    validator = context.get_validator(
        batch=batch,
        expectation_suite=suite,
    )

    # ============================================================
    # STEP 9: Add Expectations
    # ============================================================
    # Note: If the suite already exists, these expectations may be
    # duplicated on re-runs. For production, consider a dedicated
    # expectation management strategy.
    validator.expect_column_values_to_be_between("class_id", 0, 6)
    validator.expect_column_values_to_be_between("x_center", 0, 1)
    validator.expect_column_values_to_be_between("y_center", 0, 1)
    validator.expect_column_values_to_be_between("width", 0, 1)
    validator.expect_column_values_to_be_between("height", 0, 1)
    validator.expect_column_values_to_not_be_null("class_id")

    # ============================================================
    # STEP 10: Run Validation
    # ============================================================
    results = validator.validate()

    if not results.success:
        print("Great Expectations validation failed:")
        for res in results.results:
            if not res.success:
                exp_type = getattr(res.expectation_config, "type", "unknown")
                print(f"  - {exp_type}")
        return False

    print("Great Expectations validation passed")
    return True


if __name__ == "__main__":
    DATA_ROOT = os.getenv("DATA_ROOT", "data/processed")

    print(f"Validating data at: {DATA_ROOT}")
    print("=" * 50)

    success = validate_data_with_great_expectations(DATA_ROOT)

    if not success:
        print("Pipeline will stop due to validation failure.")
        sys.exit(1)  # Non-zero exit fails the CI job

    print("Data validation complete. Proceeding to training.")