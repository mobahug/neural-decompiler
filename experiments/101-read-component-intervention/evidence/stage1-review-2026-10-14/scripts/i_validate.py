"""The runner's `validate` phase (allowed by the brief), run under the review guard (no model, no forward, no write)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import guard  # noqa: E402  (first)

guard.self_test()
import importlib.util  # noqa: E402
import json  # noqa: E402

spec = importlib.util.spec_from_file_location("experiment_101_runner", guard.ROOT / "experiments/101-read-component-intervention/run.py")
runner_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner_module
spec.loader.exec_module(runner_module)
code = runner_module.main(["validate"])
print("validate exit", code)
print("guard", guard.summary())
(guard.OUT / "i_validate.json").write_text(json.dumps({"exit": code, "guard": guard.summary()}, indent=1) + "\n")
