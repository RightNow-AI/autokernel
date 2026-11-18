import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class ProfileModuleCompatibilityTest(unittest.TestCase):
    def run_python(self, code):
        return subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

    def test_profile_exposes_stdlib_api_without_project_imports(self):
        result = self.run_python(
            "import sys, profile; "
            "assert callable(profile.run); "
            "assert callable(profile.runctx); "
            "assert profile.Profile; "
            "assert profile._Utils; "
            "assert 'torch' not in sys.modules"
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cprofile_runs_from_repo_root(self):
        result = self.run_python("import cProfile; cProfile.run('sum(range(10))')")

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
