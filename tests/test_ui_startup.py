"""Each UI script must start in a FRESH interpreter, the way `streamlit run` starts it.

In-process AppTest can't catch import-path bugs: pytest has already imported `app.*`,
so a broken sys.path goes unnoticed. Here each script runs in a new Python process
from an unrelated working directory, with the script's own folder first on sys.path
(exactly what Streamlit does), so module-name shadowing and bad path inserts surface.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

UI = Path(__file__).resolve().parent.parent / "ui"

RUNNER = """
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[1])))  # what `streamlit run` does
from streamlit.testing.v1 import AppTest
at = AppTest.from_file(sys.argv[1], default_timeout=60).run()
errors = [e.value for e in at.exception]
print("EXCEPTIONS:", errors)
sys.exit(1 if errors else 0)
"""


def _entrypoints() -> list[str]:
    """The app entrypoint (whatever it's called) plus every page under views/."""
    entry = [p for p in UI.glob("*.py") if "st.navigation(" in p.read_text()]
    assert len(entry) == 1, f"expected exactly one navigation entrypoint in ui/, found {entry}"
    return [str(entry[0]), *map(str, sorted((UI / "views").glob("*.py")))]


@pytest.mark.parametrize("script", _entrypoints(), ids=lambda p: str(Path(p).relative_to(UI)))
def test_script_starts_in_a_fresh_interpreter(script, tmp_path):
    env = {
        **os.environ,
        "OPENAI_API_KEY": "sk-test",  # nothing here calls the API
        "CHROMA_PERSIST_DIR": str(tmp_path / "chroma"),
        "EVALS_DIR": str(tmp_path / "evals"),
    }
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", RUNNER, script], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
