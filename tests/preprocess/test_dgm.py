from pathlib import Path

from typer.testing import CliRunner

from seq2rel_ds import dgm
from seq2rel_ds.common.testing import Seq2RelDSTestCase

runner = CliRunner()


class TestDGM(Seq2RelDSTestCase):
    def setup_method(self) -> None:
        super().setup_method()
        self.data_dir = self.FIXTURES_ROOT / "preprocess" / "dgm"

        self.train = [
            (
                "Non-small cell lung cancer ( NSCLC ) is a common and rapidly fatal cancer for"
                " which targeted therapies have been markedly effective in about 20 % of patients ,"
                " specifically those with EGFR mutations , ROS1 rearrangements , or EML4-ALK"
                " translocations . However , only a minority of the remaining 80 % of patients"
                " likely have targetable , activating kinase mutations or translocations , and"
                " there is a great need to identify additional effective therapies [ 1 ] . We"
                " previously identified a patient with stage IV NSCLC harboring a novel BRAF"
                " mutation ( Y472C ) that had a near complete radiographic response to the"
                " multitargeted kinase inhibitor dasatinib as the sole therapy ; the patient"
                " lived without active cancer for 7 years following treatment [ 2 ] . We discovered"
                " that Y472CBRAF is a kinase inactivating BRAF mutation ( KIBRAF ) and that NSCLC"
                " cells that harbor KIBRAF undergo senescence when exposed to dasatinib , whereas"
                " NSCLC with wild-type BRAF ( WTBRAF ) or kinase activating mutations is resistant"
                " to dasatinib in vitro and in patients [ 3 ] ."
                "\tdasatinib @DRUG@ braf @GENE@ y472c @VARIANT@ @DGM@"
            ),
            (
                "A SNP causes an isoleucine to valine substitution at amino-acid codon 105 ( I105V"
                " ) in the GSTP1 gene . The valine allele occurs at a frequency of 33 % in"
                " Caucasian populations , and is associated with reduced GSTP1 activity compared to"
                " the isoleucine allele ( Watson et al , 1998 ) . This SNP has been correlated with"
                " response to cyclophosphamide chemotherapy treatment in breast cancer patients ("
                " Sweeney et al , 2000 ) . In all , 240 patients treated with cyclophosphamide were"
                " characterised for the GSTP1 I105V SNP . Patients homozygous for the valine ( low"
                " activity ) allele had a 0.3 hazard ratio ( 95 % confidence interval 0.1–1.0 ) for"
                " overall survival compared to patients homozygous for the isoleucine ( high"
                " activity ) allele . Patients heterozygous for the SNP had an intermediate hazard"
                " ratio for overall survival ( 0.8 ; 95 % confidence interval 0.5–1.3 ) ( Sweeney"
                " et al , 2000 ) . In addition , a study of 107 patients with advanced colorectal"
                " cancer treated with a combination of 5FU and oxaliplatin were assessed for the"
                " GSTP1 I105V SNP ( Stoehlmacher et al , 2002 ) . Patients homozygous for the"
                " valine allele had a median of 24.9 months survival , compared to 7.9 months for"
                " patients homozygous for the isoleucine allele ( P < 0.001 ) ( Stoehlmacher et al"
                " , 2002 ) .\t"
            ),
        ]

        self.test = [
            (
                "The EGFR-TK specific small molecule inhibitor ( TKI ) of Iressa ( Gefitinib ) was"
                " approved by FDA ( U.S Food and Drug Administration ) for advanced non-small cell"
                " lung cancer ( NSCLC ) treatment in May 2003 [ 4 ] . But the application of"
                " gefitinib suggested that just 10 % -15 % of the patients presented significant"
                " response [ 5 ] . Further studies revealed that just the tumor cell with EGFR-TK"
                " mutation ( L858R , del742-759 ) could match good response to gefitinib [ 6,7 ] ."
                "\tiressa ; gefitinib @DRUG@ egfr-tk @GENE@ l858r @VARIANT@ @DGM@"
            )
        ]

    def test_dgm_command(self, tmp_path: Path) -> None:
        input_dir = str(self.data_dir)
        output_dir = str(tmp_path)
        # The validation data is selected randomly from the train set, so turn if off
        # here to avoid the test failing due to randomness.
        result = runner.invoke(dgm.app, [input_dir, output_dir, "--valid-size", 0.0])
        assert result.exit_code == 0

        # Check that the expected files were created
        assert (tmp_path / "train.tsv").is_file()
        assert not (tmp_path / "valid.tsv").is_file()
        assert (tmp_path / "test.tsv").is_file()

        # The logic to seperate train/validation/test data is not trival, so
        # instead of testing the _preprocess method directly, we will test the
        # content of the created files.
        actual = (tmp_path / "train.tsv").read_text().strip("\n").split("\n")
        assert actual == self.train
        actual = (tmp_path / "test.tsv").read_text().strip("\n").split("\n")
        assert actual == self.test

    def test_dgm_command_pipline_entity_hinting(self, tmp_path: Path) -> None:
        input_dir = str(self.data_dir)
        output_dir = str(tmp_path)
        result = runner.invoke(dgm.app, [input_dir, output_dir])
        assert result.exit_code == 0

        # Check that the expected files were created
        assert (tmp_path / "train.tsv").is_file()
        assert (tmp_path / "valid.tsv").is_file()
        assert (tmp_path / "test.tsv").is_file()

    def test_dgm_command_gold_entity_hinting(self, tmp_path: Path) -> None:
        input_dir = str(self.data_dir)
        output_dir = str(tmp_path)
        result = runner.invoke(dgm.app, [input_dir, output_dir])
        assert result.exit_code == 0

        # Check that the expected files were created
        assert (tmp_path / "train.tsv").is_file()
        assert (tmp_path / "valid.tsv").is_file()
        assert (tmp_path / "test.tsv").is_file()
