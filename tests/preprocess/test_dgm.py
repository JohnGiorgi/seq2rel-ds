from pathlib import Path

from seq2rel_ds.common.testing import Seq2RelDSTestCase
from seq2rel_ds.preprocess import dgm
from typer.testing import CliRunner

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
                "As stromal cells often reside in , or are recruited to the vicinity of the tumor ,"
                " we sought to establish an in vitro co-cultivation model that recapitulates this"
                " encounter and permits an efficient separation and characterization of the two"
                " cell populations . As we planned to investigate the effect of mutant p53 , we"
                " chose to work with lung cancer cells ( H1299 ) which are null for p53 expression"
                " and introduced them with two p53 ‘hotsopt’ mutations residing within the DNA"
                " binding domain , namely R175H and R248Q ( H1299175 and H1299248 respectively ,"
                " Figure 1A and B ) . The cells were then labeled with a red fluorescent protein"
                " ( dsRed ) , while lung CAFs ( HK3-T ) were labeled with a green fluorescent"
                " protein ( GFP ) . The labeled populations were co-cultivated for 24 hours and"
                " separated by Fluorescence Associated Cell Sorting ( FACS ) based on their"
                " specific fluorescent marker ( Figure 1C ) . To minimize the possibility of cross"
                " contamination , the separated populations were sorted again , and indeed , the"
                " level of cross contamination was diminished ( Figure 1C ) . To further"
                " corroborate this observation , we also performed quantitative real time PCR"
                " ( QRT-PCR ) with primers amplifying either GFP or dsRed . Following the double"
                " sorting procedure , GFP and dsRed expression was several orders of magnitude"
                " higher in the corresponding labeled cells ( Figure S1A ) . Because the sorting"
                " procedure includes prolonged incubation on ice , and cells may be subjected to"
                " mechanical stress introduced by the FACS machinery , we decided to measure the"
                " expression levels of stress related genes prior and post the sorting procedure ."
                " First , p21 , a common stress response gene , was found to be expressed in a"
                " comparable manner in the sorted and unsorted samples ( Figure S1B ) . Moreover ,"
                " several other genes that are known to be specifically elevated during mechanical"
                " stress in lung cells [ 25 ] were found to be either equally expressed or down"
                " regulated following the sorting procedure ( Figure S1B ) ."
                " 10.1371/journal.pone.0061353.g001Figure 1 An in vitro model to study the"
                " tumor-stroma encounter in lung cancer . ( A ) . p53-null lung carcinoma cells"
                " ( H1299 ) were introduced with the designated mutations . p53 levels were"
                " determined by Western blot analysis ( A ) and by QRT-PCR ( B ) . A fluorescent"
                " microscope image of co-cultured dsRed labeled H1299 with GFP labeled HK3 ( C ,"
                " upper panel ) . Representative FACS analysis depicting dsRed- and GFP labeled"
                " sub-populations following a sorting procedure ( C , middle panel ) ."
                " Each sub-population was then re-sorted using the same sorting gates ( C , lower"
                " panel ) .\t"
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
