from pathlib import Path

from seq2rel_ds.common.testing import Seq2RelDSTestCase
from seq2rel_ds.preprocess import ade
from typer.testing import CliRunner
from seq2rel_ds.common import util

runner = CliRunner()


class TestADE(Seq2RelDSTestCase):
    def test_preprocess(self) -> None:
        # In absense of testing the entire data set, we sanity check the first and last examples
        actual = ade._preprocess()
        assert actual[0] == (
            "Intravenous azithromycin-induced ototoxicity."
            f"\t@{ade.REL_LABEL}@ azithromycin @{ade.DRUG_LABEL}@ ototoxicity @{ade.EFFECT_LABEL}@ {util.END_OF_REL_SYMBOL}"
        )
        assert actual[-1] == (
            "Successful challenge with clozapine in a history of eosinophilia."
            f"\t@{ade.REL_LABEL}@ clozapine @{ade.DRUG_LABEL}@ eosinophilia @{ade.EFFECT_LABEL}@ {util.END_OF_REL_SYMBOL}"
        )

    def test_ade_command(self, tmp_path: Path) -> None:
        output_dir = str(tmp_path)
        result = runner.invoke(ade.app, [output_dir])
        assert result.exit_code == 0

        # Check that the expected files were created
        assert (tmp_path / "train.tsv").is_file()
        assert (tmp_path / "valid.tsv").is_file()
        assert (tmp_path / "test.tsv").is_file()
