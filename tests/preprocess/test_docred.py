import json
from pathlib import Path

from seq2rel_ds.common.testing import Seq2RelDSTestCase
from seq2rel_ds.preprocess import docred
from typer.testing import CliRunner

runner = CliRunner()


class TestDocRED(Seq2RelDSTestCase):
    def setup_method(self) -> None:
        super().setup_method()
        self.data_dir = self.FIXTURES_ROOT / "preprocess" / "DocRED"
        self.train_path = self.data_dir / docred.TRAIN_ANNOTATED_FILENAME
        self.valid_path = self.data_dir / docred.VALID_FILENAME
        self.test_path = self.data_dir / docred.TEST_FILENAME
        # This is the dictionary mapping rel ids to rel labels provided by DocRED
        self.rel_info = json.loads((self.data_dir / docred.REL_INFO_FILENAME).read_text())

        # The expected data, after preprocessing, for each partition
        self.train = [
            (
                "Zest Airways , Inc. operated as AirAsia Zest ( formerly Asian Spirit and Zest Air"
                " ) , was a low - cost airline based at the Ninoy Aquino International Airport in"
                " Pasay City , Metro Manila in the Philippines . It operated scheduled domestic and"
                " international tourist services , mainly feeder services linking Manila and Cebu"
                " with 24 domestic destinations in support of the trunk route operations of other"
                " airlines . In 2013 , the airline became an affiliate of Philippines AirAsia"
                " operating their brand separately . Its main base was Ninoy Aquino International"
                " Airport , Manila . The airline was founded as Asian Spirit , the first airline in"
                " the Philippines to be run as a cooperative . On August 16 , 2013 , the Civil"
                " Aviation Authority of the Philippines ( CAAP ) , the regulating body of the"
                " Government of the Republic of the Philippines for civil aviation , suspended Zest"
                " Air flights until further notice because of safety issues . Less than a year"
                " after AirAsia and Zest Air 's strategic alliance , the airline has been rebranded"
                " as AirAsia Zest . The airline was merged into AirAsia Philippines in January 2016 ."
                "\t@HEADQUARTERS_LOCATION@ zest airways, inc.; asian spirit and zest air; airasia zest @ORG@ pasay city @LOC@ @EOR@"
                " @COUNTRY@ zest airways, inc.; asian spirit and zest air; airasia zest @ORG@ philippines; republic of the philippines @LOC@ @EOR@"
                " @COUNTRY@ asian spirit @ORG@ philippines; republic of the philippines @LOC@ @EOR@"
                " @LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY@ ninoy aquino international airport @LOC@ pasay city @LOC@ @EOR@"
                " @COUNTRY@ ninoy aquino international airport @LOC@ philippines; republic of the philippines @LOC@ @EOR@"
                " @COUNTRY@ zest air @ORG@ philippines; republic of the philippines @LOC@ @EOR@"
                " @COUNTRY@ manila @LOC@ philippines; republic of the philippines @LOC@ @EOR@"
                " @CONTAINS_ADMINISTRATIVE_TERRITORIAL_ENTITY@ philippines; republic of the philippines @LOC@ metro manila @LOC@ @EOR@"
                " @COUNTRY@ pasay city @LOC@ philippines; republic of the philippines @LOC@ @EOR@"
                " @LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY@ pasay city @LOC@ metro manila @LOC@ @EOR@"
                " @CONTAINS_ADMINISTRATIVE_TERRITORIAL_ENTITY@ metro manila @LOC@ pasay city @LOC@ @EOR@"
                " @LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY@ metro manila @LOC@ philippines; republic of the philippines @LOC@ @EOR@"
                " @COUNTRY@ metro manila @LOC@ philippines; republic of the philippines @LOC@ @EOR@"
            ),
            (
                "The short - beaked common dolphin ( Delphinus delphis ) is a species of common"
                " dolphin . It has a larger range than the long - beaked common dolphin ( D."
                " capensis ) , occurring throughout warm - temperate and tropical oceans ,"
                " including the Indian Ocean although in smaller quantities than other places they"
                " are found . There are more short - beaked common dolphins than any other dolphin"
                " species in the warm - temperate portions of the Atlantic and Pacific Oceans . It"
                " is also found in the Caribbean and Mediterranean Seas . The short - beaked common"
                " dolphin is also abundant in the Black Sea , Gulf of Mexico , and Red Sea . They"
                " follow the gulf stream up to Norwegian waters . Seldom do any short - beaked"
                " dolphin venture near the Arctic .\t"
            ),
        ]
        self.valid = [
            (
                "Lark Force was an Australian Army formation established in March 1941 during World"
                " War II for service in New Britain and New Ireland . Under the command of"
                " Lieutenant Colonel John Scanlan , it was raised in Australia and deployed to"
                " Rabaul and Kavieng , aboard SS Katoomba , MV Neptuna and HMAT Zealandia , to"
                " defend their strategically important harbours and airfields . The objective of"
                " the force , was to maintain a forward air observation line as long as possible"
                " and to make the enemy fight for this line rather than abandon it at the first"
                " threat as the force was considered too small to withstand any invasion . Most of"
                " Lark Force was captured by the Imperial Japanese Army after Rabaul and Kavieng"
                " were captured in January 1942 . The officers of Lark Force were transported to"
                " Japan , however the NCOs and men were unfortunately torpedoed by the USS Sturgeon"
                " while being transported aboard the Montevideo Maru . Only a handful of the"
                " Japanese crew were rescued , with none of the between 1,050 and 1,053 prisoners"
                " aboard surviving as they were still locked below deck ."
                "\t@OPERATOR@ lark force @ORG@ australian army @ORG@ @EOR@"
                " @INCEPTION@ lark force @ORG@ march 1941 @TIME@ @EOR@"
                " @CONFLICT@ lark force @ORG@ world war ii @MISC@ @EOR@"
                " @COUNTRY@ lark force @ORG@ australia @LOC@ @EOR@"
                " @CONFLICT@ australian army @ORG@ world war ii @MISC@ @EOR@"
                " @COUNTRY@ australian army @ORG@ australia @LOC@ @EOR@"
                " @MILITARY_BRANCH@ john scanlan @PER@ australian army @ORG@ @EOR@"
                " @CONFLICT@ john scanlan @PER@ world war ii @MISC@ @EOR@"
                " @COUNTRY_OF_CITIZENSHIP@ john scanlan @PER@ australia @LOC@ @EOR@"
                " @PARTICIPANT_OF@ japan @LOC@ world war ii @MISC@ @EOR@"
                " @ETHNIC_GROUP@ japan @LOC@ japanese @LOC@ @EOR@"
                " @CONFLICT@ imperial japanese army @ORG@ world war ii @MISC@ @EOR@"
                " @COUNTRY@ imperial japanese army @ORG@ japan @LOC@ @EOR@"
                " @COUNTRY@ imperial japanese army @ORG@ japanese @LOC@ @EOR@"
                " @PARTICIPANT_OF@ australia @LOC@ world war ii @MISC@ @EOR@"
                " @CONFLICT@ uss sturgeon @MISC@ world war ii @MISC@ @EOR@"
                " @COUNTRY@ mv neptuna @MISC@ australia @LOC@ @EOR@"
                " @COUNTRY@ hmat zealandia @MISC@ australia @LOC@ @EOR@"
            ),
            (
                "The 4th House of Orléans , sometimes called the House of Bourbon - Orléans ( ) to"
                " distinguish it , is the fourth holder of a surname previously used by several"
                " branches of the Royal House of France , all descended in the legitimate male line"
                " from the dynasty 's founder , Hugh Capet . The house was founded in 1661 by"
                " Prince Philippe , Duke of Anjou , younger son of king Louis XIII and younger"
                ' brother of king Louis XIV , the " Sun King " . From 1709 until the French'
                " Revolution , the Orléans dukes were next in the order of succession to the French"
                " throne after members of the senior branch of the House of Bourbon , descended"
                " from king Louis XIV . Although Louis XIV 's direct descendants retained the"
                " throne , his brother Philippe 's descendants flourished until the end of the"
                " French monarchy . They held the Crown from 1830 to 1848 , and they still are"
                " pretenders to the French throne ."
                "\t@ETHNIC_GROUP@ louis xiv; sun king @PER@ french @LOC@ @EOR@"
                " @LANGUAGES_SPOKEN_WRITTEN_OR_SIGNED@ louis xiv; sun king @PER@ french @LOC@ @EOR@"
                " @FATHER@ louis xiv; sun king @PER@ louis xiii @PER@ @EOR@"
                " @CHILD@ louis xiii @PER@ louis xiv; sun king @PER@ @EOR@"
            ),
        ]
        self.test = [
            (
                "Miguel Riofrio Sánchez ( September 7 , 1822 – October 11 , 1879 ) was an Ecuadoran"
                " poet , novelist , journalist , orator , and educator . He was born in the city of"
                " Loja . He is best known today as the author of Ecuador 's first novel La"
                " Emancipada ( 1863 ) . Owing to the book 's length , usually less than 100 pages"
                " long , many experts have argued that it is really a novella rather than a full"
                " novel , and that Ecuador 's first novel is Juan León Mera 's Cumanda ( 1879 ) ."
                " Nevertheless , thanks to the arguments of the well - known and respected"
                " Ecuadorian writer Alejandro Carrión ( 1915 – 1992 ) , Miguel Riofrío 's La"
                " Emancipada has been accepted as Ecuador 's first novel . Riofrio died in exile in"
                " Peru .\t"
            )
        ]

    def test_preprocess_docred(self) -> None:
        # training data
        actual = docred._preprocess(self.train_path, rel_labels=self.rel_info)
        assert actual == self.train

        # validation data
        actual = docred._preprocess(self.valid_path, rel_labels=self.rel_info)
        assert actual == self.valid

        # test data
        actual = docred._preprocess(self.test_path)
        assert actual == self.test

    def test_docred_command(self, tmp_path: Path) -> None:

        input_dir = str(self.data_dir)
        output_dir = str(tmp_path)
        result = runner.invoke(docred.app, [input_dir, output_dir])
        assert result.exit_code == 0

        # training data
        actual = (tmp_path / "train.tsv").read_text().strip("\n").split("\n")
        assert actual == self.train

        # validation data
        actual = (tmp_path / "valid.tsv").read_text().strip("\n").split("\n")
        assert actual == self.valid

        # test data
        actual = (tmp_path / "test.tsv").read_text().strip("\n").split("\n")
        assert actual == self.test
