from pathlib import Path

from seq2rel_ds.common.testing import Seq2RelDSTestCase
from seq2rel_ds.preprocess import bc5cdr
from typer.testing import CliRunner

runner = CliRunner()


class TestBC5CDR(Seq2RelDSTestCase):
    def setup_method(self) -> None:
        super().setup_method()
        self.data_dir = self.FIXTURES_ROOT / "preprocess" / "bc5cdr"
        self.train_path = self.data_dir / bc5cdr.TRAIN_FILENAME
        self.valid_path = self.data_dir / bc5cdr.VALID_FILENAME
        self.test_path = self.data_dir / bc5cdr.TEST_FILENAME

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
                "\t@CID@ alpha-methyldopa @CHEMICAL@ hypotensive @DISEASE@ @EOR@"
            ),
            (
                "Lidocaine-induced cardiac asystole. Intravenous administration of a single 50-mg bolus"
                " of lidocaine in a 67-year-old man resulted in profound depression of the activity of"
                " the sinoatrial and atrioventricular nodal pacemakers. The patient had no apparent"
                " associated conditions which might have predisposed him to the development of"
                " bradyarrhythmias; and, thus, this probably represented a true idiosyncrasy to"
                " lidocaine."
                "\t@CID@ lidocaine @CHEMICAL@ cardiac asystole @DISEASE@ @EOR@"
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
                "\t@CID@ lithium carbonate @CHEMICAL@ neurologic depression @DISEASE@ @EOR@"
                " @CID@ lithium carbonate @CHEMICAL@ cyanosis @DISEASE@ @EOR@"
                " @CID@ lithium carbonate @CHEMICAL@ cardiac arrhythmia @DISEASE@ @EOR@"
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
                "\t@CID@ famotidine @CHEMICAL@ delirium @DISEASE@ @EOR@"
            )
        ]

    def test_preprocess_bc5cdr(self) -> None:
        # training data
        actual = bc5cdr._preprocess_bc5cdr(self.train_path)
        assert actual == self.train

        # validation data
        actual = bc5cdr._preprocess_bc5cdr(self.valid_path)
        assert actual == self.valid

        # test data
        actual = bc5cdr._preprocess_bc5cdr(self.test_path)
        assert actual == self.test

    def test_bc5cdr_command(self, tmp_path: Path) -> None:

        input_dir = str(self.data_dir)
        output_dir = str(tmp_path)
        result = runner.invoke(bc5cdr.app, [input_dir, output_dir])
        assert result.exit_code == 0

        # training data
        actual = (tmp_path / "train.tsv").read_text().strip().split("\n")
        assert actual == self.train

        # validation data
        actual = (tmp_path / "valid.tsv").read_text().strip().split("\n")
        assert actual == self.valid

        # test data
        actual = (tmp_path / "test.tsv").read_text().strip().split("\n")
        assert actual == self.test
