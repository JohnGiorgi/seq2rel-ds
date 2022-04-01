from pathlib import Path

from typer.testing import CliRunner

from seq2rel_ds import gda
from seq2rel_ds.common.testing import Seq2RelDSTestCase

runner = CliRunner()


class TestGDA(Seq2RelDSTestCase):
    def setup_method(self) -> None:
        super().setup_method()
        self.data_dir = self.FIXTURES_ROOT / "preprocess" / "gda"
        self.train_path = self.data_dir / gda.TRAIN_DATA
        self.test_path = self.data_dir / gda.TEST_DATA

        # The expected data, after preprocessing, for each partition
        self.train = [
            (
                "The same molecular defects of the gonadotropin-releasing hormone receptor determine"
                " a variable degree of hypogonadism in affected kindred. Detailed endocrinological"
                " studies were performed in the three affected kindred of a family carrying mutations"
                " of the GnRH receptor gene. All three were compound heterozygotes carrying on one"
                " allele the Arg262Gln mutation and on the other allele two mutations (Gln106Arg and"
                " Ser217Arg). When expressed in heterologous cells, both Gln106Arg and Ser217Arg"
                " mutations altered hormone binding, whereas the Arg262Gln mutation altered activation"
                " of phospholipase C. The propositus, a 30-yr-old man, displayed complete idiopathic"
                " hypogonadotropic hypogonadism with extremely low plasma levels of gonadotropins,"
                " absence of pulsatility of endogenous LH and alpha-subunit, absence of response to"
                " GnRH and GnRH agonist (triptorelin), and absence of effect of pulsatile"
                " administration of GnRH. The two sisters, 24 and 18 yr old, of the propositus"
                " displayed, on the contrary, only partial idiopathic hypogonadotropic hypogonadism."
                " They both had primary amenorrhea, and the younger sister displayed retarded bone"
                " maturation and uterus development, but both sisters had normal breast development."
                " Gonadotropin concentrations were normal or low, but in both cases were restored to"
                " normal levels by a single injection of GnRH. In the two sisters, there were no"
                " spontaneous pulses of LH, but pulsatile administration of GnRH provoked a pulsatile"
                " secretion of LH in the younger sister. The same mutations of the GnRH receptor gene"
                " may thus determine different degrees of alteration of gonadotropin function in"
                " affected kindred of the same family."
                "\tgonadotropin-releasing hormone receptor ; gnrh receptor @GENE@ complete"
                " idiopathic hypogonadotropic hypogonadism ; idiopathic hypogonadotropic hypogonadism"
                " @DISEASE@ @GDA@"
            ),
            (
                "The NAT2* slow acetylator genotype is associated with bladder cancer in Taiwanese,"
                " but not in the Black Foot Disease endemic area population."
                "\tnat2 @GENE@ bladder cancer @DISEASE@ @GDA@"
            ),
        ]
        self.test = [
            (
                "Variations in the monoamine oxidase B (MAOB) gene are associated with Parkinson's"
                " disease. The monoamine oxidase B gene (MAOB; Xp15.21-4) is a candidate gene for"
                " Parkinson's disease (PD) given its role in dopamine metabolism and its possible"
                " role in the activation of neurotoxins. The association of MAOB polymorphisms"
                " (a [GT] repeat allelic variation in intron 2 and an A-G transition in intron 13)"
                " with Parkinson's disease (PD) was studied in an Australian cohort of 204"
                " (male:female ratio 1.60) people with PD and 285 (male:female ratio 1.64)"
                " age- and gender-matched control subjects. Genomic DNA was extracted from venous"
                " blood and polymerase chain reaction was used to amplify the appropriate regions"
                " of the MAOB gene. The length of each (GT) repeat sequence was determined by 5%"
                " polyacrylamide denaturing gel electrophoresis and a DNA fragment analyzer, while"
                " the G-A genotype was determined using 2% agarose gel electrophoresis. The G-A"
                " polymorphism showed no association with PD (odds ratio [OR] = 0.80; p = 0.51;"
                " 95% confidence interval [CI] = 0.42-1.53). There was a significant difference in"
                " allele frequencies of the (GT) repeat allelic variation between patients and"
                " control subjects (chi2 = 20.09; p<0.01). After statistical adjustment for"
                " potential confounders using a logistic regression analysis, the (GT) repeat"
                " alleles > or =188 base pairs in the intron 2 marker of the MAOB gene were"
                " significantly associated with PD (OR = 4.60; p<0.00005; 95% CI = 1.97-10.77)."
                " The 186 base pair allele was also significantly associated with PD (OR = 1.85;"
                " p = 0.048; 95% CI = 1.01-3.42). The GT repeat in intron 2 of the MAOB gene is a"
                " powerful marker for PD in this large Australian cohort."
                "\tmonoamine oxidase b ; maob @GENE@ parkinson's disease ; pd @DISEASE@ @GDA@"
            )
        ]

    def test_preprocess(self) -> None:
        # training data
        abstracts = (self.train_path / gda.ABSTRACTS_FILENAME).read_text()
        anns = (self.train_path / gda.ANNS_FILENAME).read_text()
        labels = (self.train_path / gda.LABELS_FILENAME).read_text()
        train_raw = [abstracts, anns, labels]
        actual = gda._preprocess(train_raw)
        assert actual == self.train

        # test data
        abstracts = (self.test_path / gda.ABSTRACTS_FILENAME).read_text()
        anns = (self.test_path / gda.ANNS_FILENAME).read_text()
        labels = (self.test_path / gda.LABELS_FILENAME).read_text()
        test_raw = [abstracts, anns, labels]
        actual = gda._preprocess(test_raw)
        assert actual == self.test

    def test_gda_command(self, tmp_path: Path) -> None:
        output_dir = str(tmp_path)
        result = runner.invoke(gda.app, [output_dir])
        assert result.exit_code == 0

        # Check that the expected files were created
        assert (tmp_path / "train.tsv").is_file()
        assert (tmp_path / "valid.tsv").is_file()
        assert (tmp_path / "test.tsv").is_file()

    def test_gda_command_pipline_entity_hinting(self, tmp_path: Path) -> None:
        output_dir = str(tmp_path)
        result = runner.invoke(gda.app, [output_dir, "--entity-hinting", "pipeline"])
        assert result.exit_code == 0

        # Check that the expected files were created
        assert (tmp_path / "train.tsv").is_file()
        assert (tmp_path / "valid.tsv").is_file()
        assert (tmp_path / "test.tsv").is_file()

    def test_gda_command_gold_entity_hinting(self, tmp_path: Path) -> None:
        output_dir = str(tmp_path)
        result = runner.invoke(gda.app, [output_dir, "--entity-hinting", "gold"])
        assert result.exit_code == 0

        # Check that the expected files were created
        assert (tmp_path / "train.tsv").is_file()
        assert (tmp_path / "valid.tsv").is_file()
        assert (tmp_path / "test.tsv").is_file()
