import pathlib


class Seq2RelDSTestCase:
    """A custom testing case that contains any logic that might be shared across tests"""

    PROJECT_ROOT = (pathlib.Path(__file__).parent / ".." / ".." / "..").resolve()
    MODULE_ROOT = PROJECT_ROOT / "seq2rel_ds"
    TESTS_ROOT = PROJECT_ROOT / "tests"
    FIXTURES_ROOT = PROJECT_ROOT / "test_fixtures"

    def setup_method(self):
        pass
