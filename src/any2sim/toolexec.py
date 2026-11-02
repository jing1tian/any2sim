"""Entry point inside a worker env: python -m any2sim.toolexec <tool> <project> <params-json> <result-path>."""
import importlib
import json
import sys
import traceback
from pathlib import Path


def main():
    name, project, params, result_path = sys.argv[1:5]
    try:
        mod = importlib.import_module(f"any2sim.tools.{name}")
        result = mod.run(Path(project), **json.loads(params))
    except Exception as e:  # the receipt records the failure; the traceback goes to the log
        traceback.print_exc()
        result = {"error": f"{type(e).__name__}: {e}"}
    from any2sim.common import write_json
    write_json(result, result_path)
    sys.exit(0 if "error" not in result else 1)


if __name__ == "__main__":
    main()
