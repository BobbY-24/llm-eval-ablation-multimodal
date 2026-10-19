"""Run local fixture tests with finite CPU/output/address-space allowances."""
import os
import json
import platform
import resource
import sys
import unittest
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024**2, 64 * 1024**2))
# macOS does not reliably enforce RLIMIT_AS for all native allocation paths;
# finite parser/decoder limits and the OS filesystem/network sandbox also apply.
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
suite = (unittest.defaultTestLoader.loadTestsFromNames(sys.argv[1:]) if len(sys.argv) > 1
         else unittest.defaultTestLoader.discover(str(ROOT / "tests")))
result = unittest.TextTestRunner(verbosity=2).run(suite)
versions = {}
for name in ("numpy", "Pillow", "av", "h5py", "peppi-py", "PyYAML"):
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        versions[name] = None
report = {"evidence_type": "local_fixture_conformance_not_real_dataset_validation",
          "completed_utc": datetime.now(timezone.utc).isoformat(),
          "python": platform.python_version(), "optional_dependencies": versions,
          "selection": sys.argv[1:] or ["all"], "tests_run": result.testsRun,
          "passed": result.testsRun - len(result.errors) - len(result.failures) - len(result.skipped),
          "errors": [{"test": str(test), "traceback": trace} for test, trace in result.errors],
          "failures": [{"test": str(test), "traceback": trace} for test, trace in result.failures],
          "skipped": [{"test": str(test), "reason": reason} for test, reason in result.skipped],
          "successful": result.wasSuccessful(), "real_dataset_samples_validated": 0,
          "isolation": "macOS sandbox-exec wrapper; empty environment; network denied; bounded CPU/file output"}
destination = ROOT / "artifacts" / ("verification-python-" + platform.python_version() + ".json")
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
raise SystemExit(not result.wasSuccessful())
