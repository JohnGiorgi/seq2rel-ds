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
                "The present study clearly shows that both ponatinib and panobinostat showed"
                " antiproliferative effects and cytotoxicity against the IM‐resistant variant"
                " K562/IM‐R1 cells with BCR‐ABL amplification and Ba/F3/T315I cells with BCR‐ABL"
                " with a T315I kinase mutation ( Figs 2 , 3 , 4d ) . Moreover , the combination of"
                " panobinostat and ponatinib showed synergistic antiproliferative and cytotoxic"
                " effects on these cell lines ( Fig. 3 ) . Thus , ponatinib inhibited the"
                " constitutive ABL kinase activity in cells expressing gene‐amplified active ABL or"
                " in cells expressing ABL with a T315I mutation ( Fig. 5b–e ) . This inhibition of"
                " ABL kinase activity was accompanied by inhibition of the phosphorylation of"
                " downstream signaling molecules ."
                "\t@DGM@ ponatinib @DRUG@ abl @GENE@ t315i @VARIANT@ @EOR@"
            )
        ]

        self.valid = [
            (
                "Another mechanism underlying imatinib resistance is point mutations in the BCR-ABL"
                " gene altering the conformation of the ATP binding pocket such that imatinib no longer"
                " has affinity . Analysis of the 579-base pair region , which gives rise to the ATP"
                " binding pocket and the activation loop of the kinase domain on BCR-ABL , revealed in"
                " six of nine patients a C→T change at nucleotide 944 ( Gorre et al , 2001 ) ."
                " The latter transition corresponds to an isoleucine substitution for threonine at"
                " position 315 ( T315I ) in the ATP binding pocket , which does not inhibit ATP"
                " binding . Such a change , however , would inhibit drug binding owing to steric"
                " hindrance and loss of critical hydrogen bond formation . Von Bubnoff et al ( 2002 )"
                " have analysed a group of eight patients with Ph+ leukaemias , of which two were CML"
                " in BC or AP/BC . This group also detected the T315I in one patient , as well as"
                " reporting four novel point mutations all located within the drug binding pocket on"
                " BCR-ABL. Branford et al ( 2002 ) found that nine of 12 CML patients resistant to"
                " imatinib had a mutation within the ATP binding region of BCR-ABL . A range of"
                " mutations was observed , including T315I . Roumiantsev et al ( 2002 ) found that the"
                " Y253F mutation in the Abl kinase domain conferred intermediate resistance to imatinib"
                " , both in vitro and in vivo , relative to the T315I mutation ."
                "\t@DGM@ imatinib @DRUG@ abl @GENE@ y253f @VARIANT@ @EOR@"
            )
        ]

    def test_gdm_command(self, tmp_path: Path) -> None:

        input_dir = str(self.data_dir)
        output_dir = str(tmp_path)
        result = runner.invoke(dgm.app, [input_dir, output_dir])
        assert result.exit_code == 0

        # training data
        actual = (tmp_path / "train.tsv").read_text().strip("\n").split("\n")
        assert actual == self.train

        # validation data
        actual = (tmp_path / "valid.tsv").read_text().strip("\n").split("\n")
        assert actual == self.valid
