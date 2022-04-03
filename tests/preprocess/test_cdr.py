from pathlib import Path

from typer.testing import CliRunner

from seq2rel_ds import cdr
from seq2rel_ds.common import schemas
from seq2rel_ds.common.testing import Seq2RelDSTestCase

runner = CliRunner()


class TestCDR(Seq2RelDSTestCase):
    def setup_method(self) -> None:
        super().setup_method()
        self.data_dir = self.FIXTURES_ROOT / "preprocess" / "cdr"
        self.train_path = self.data_dir / cdr.TRAIN_FILENAME
        self.valid_path = self.data_dir / cdr.VALID_FILENAME
        self.test_path = self.data_dir / cdr.TEST_FILENAME

        # The expected data, after preprocessing, for each partition
        self.train = [
            (
                "Naloxone reverses the antihypertensive effect of clonidine. In unanesthetized,"
                " spontaneously hypertensive rats the decrease in blood pressure and heart rate"
                " produced by intravenous clonidine, 5 to 20 micrograms/kg, was inhibited or reversed"
                " by nalozone, 0.2 to 2 mg/kg. The hypotensive effect of 100 mg/kg alpha-methyldopa was"
                " also partially reversed by naloxone. Naloxone alone did not affect either blood"
                " pressure or heart rate. In brain membranes from spontaneously hypertensive rats"
                " clonidine, 10(-8) to 10(-5) M, did not influence stereoselective binding of"
                " [3H]-naloxone (8 nM), and naloxone, 10(-8) to 10(-4) M, did not influence"
                " clonidine-suppressible binding of [3H]-dihydroergocryptine (1 nM). These findings"
                " indicate that in spontaneously hypertensive rats the effects of central"
                " alpha-adrenoceptor stimulation involve activation of opiate receptors. As naloxone"
                " and clonidine do not appear to interact with the same receptor site, the observed"
                " functional antagonism suggests the release of an endogenous opiate by clonidine or"
                " alpha-methyldopa and the possible role of the opiate in the central control of"
                " sympathetic tone."
                "\talpha-methyldopa @CHEMICAL@ hypotensive @DISEASE@ @CID@"
            ),
            (
                "Sulpiride-induced tardive dystonia. Sulpiride is a selective D2-receptor"
                " antagonist with antipsychotic and antidepressant properties. Although initially"
                " thought to be free of extrapyramidal side effects, sulpiride-induced tardive"
                " dyskinesia and parkinsonism have been reported occasionally. We studied a"
                " 37-year-old man who developed persistent segmental dystonia within 2 months"
                " after starting sulpiride therapy. We could not find any previous reports of"
                " sulpiride-induced tardive dystonia."
                "\tsulpiride @CHEMICAL@ tardive dystonia ; dystonia @DISEASE@ @CID@"
            ),
        ]
        self.valid = [
            (
                "Tricuspid valve regurgitation and lithium carbonate toxicity in a newborn infant."
                " A newborn with massive tricuspid regurgitation, atrial flutter, congestive heart"
                " failure, and a high serum lithium level is described. This is the first patient"
                " to initially manifest tricuspid regurgitation and atrial flutter, and the 11th"
                " described patient with cardiac disease among infants exposed to lithium compounds"
                " in the first trimester of pregnancy. Sixty-three percent of these infants had"
                " tricuspid valve involvement. Lithium carbonate may be a factor in the increasing"
                " incidence of congenital heart disease when taken during early pregnancy. It also"
                " causes neurologic depression, cyanosis, and cardiac arrhythmia when consumed"
                " prior to delivery."
                "\tlithium carbonate @CHEMICAL@ neurologic depression @DISEASE@ @CID@"
                " lithium carbonate @CHEMICAL@ cyanosis @DISEASE@ @CID@"
                " lithium carbonate @CHEMICAL@ cardiac arrhythmia @DISEASE@ @CID@"
            )
        ]
        self.test = [
            (
                "Famotidine-associated delirium. A series of six cases. Famotidine is a histamine"
                " H2-receptor antagonist used in inpatient settings for prevention of stress ulcers"
                " and is showing increasing popularity because of its low cost. Although all of the"
                " currently available H2-receptor antagonists have shown the propensity to cause"
                " delirium, only two previously reported cases have been associated with"
                " famotidine. The authors report on six cases of famotidine-associated delirium in"
                " hospitalized patients who cleared completely upon removal of famotidine. The"
                " pharmacokinetics of famotidine are reviewed, with no change in its metabolism in"
                " the elderly population seen. The implications of using famotidine in elderly"
                " persons are discussed."
                "\tfamotidine @CHEMICAL@ delirium @DISEASE@ @CID@"
            )
        ]

    def test_filter_hypernyms(self):
        annotation = schemas.PubtatorAnnotation(
            text=(
                "Carbamazepine-induced cardiac dysfunction. A patient with sinus bradycardia and"
                " atrioventricular block, induced by carbamazepine, prompted an extensive"
                " literature review of all previously reported cases."
            ),
            pmid="",
            entities={
                "D002220": schemas.PubtatorEntity(
                    mentions=["Carbamazepine", "carbamazepine"],
                    offsets=[(0, 13), (115, 128)],
                    label="Chemical",
                ),
                "D006331": schemas.PubtatorEntity(
                    mentions=["cardiac dysfunction"],
                    offsets=[(22, 41)],
                    label="Disease",
                ),
                "D001919": schemas.PubtatorEntity(
                    mentions=["bradycardia"],
                    offsets=[(64, 75)],
                    label="Disease",
                ),
                "D054537": schemas.PubtatorEntity(
                    mentions=["atrioventricular block"],
                    offsets=[(80, 102)],
                    label="Disease",
                ),
            },
            relations=[("D002220", "D001919", "CID"), ("D002220", "D054537", "CID")],
        )

        cdr._filter_hypernyms([annotation])
        actual = annotation.filtered_relations
        # D006331 is a hypernym of D001919 and/or D054537 and so it should be filtered.
        expected = [("D002220", "D006331", "CID")]

        assert actual == expected

    def test_preprocess(self) -> None:
        # training data
        actual = cdr._preprocess(self.train_path.read_text())
        assert actual == self.train

        # validation data
        actual = cdr._preprocess(self.valid_path.read_text())
        assert actual == self.valid

        # test data
        actual = cdr._preprocess(self.test_path.read_text())
        assert actual == self.test

    def test_cdr_command(self, tmp_path: Path) -> None:
        output_dir = str(tmp_path)
        result = runner.invoke(cdr.app, [output_dir])
        assert result.exit_code == 0

        # Check that the expected files were created
        assert (tmp_path / "train.tsv").is_file()
        assert (tmp_path / "valid.tsv").is_file()
        assert (tmp_path / "test.tsv").is_file()

    def test_cdr_command_pipline_entity_hinting(self, tmp_path: Path) -> None:
        output_dir = str(tmp_path)
        result = runner.invoke(cdr.app, [output_dir, "--entity-hinting", "pipeline"])
        assert result.exit_code == 0

        # Check that the expected files were created
        assert (tmp_path / "train.tsv").is_file()
        assert (tmp_path / "valid.tsv").is_file()
        assert (tmp_path / "test.tsv").is_file()

    def test_cdr_command_gold_entity_hinting(self, tmp_path: Path) -> None:
        output_dir = str(tmp_path)
        result = runner.invoke(cdr.app, [output_dir, "--entity-hinting", "gold"])
        assert result.exit_code == 0

        # Check that the expected files were created
        assert (tmp_path / "train.tsv").is_file()
        assert (tmp_path / "valid.tsv").is_file()
        assert (tmp_path / "test.tsv").is_file()
