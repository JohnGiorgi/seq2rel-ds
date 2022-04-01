import json
from pathlib import Path

from typer.testing import CliRunner

from seq2rel_ds import docred
from seq2rel_ds.common.testing import Seq2RelDSTestCase

runner = CliRunner()


class TestDocRED(Seq2RelDSTestCase):
    def setup_method(self) -> None:
        super().setup_method()
        self.data_dir = self.FIXTURES_ROOT / "preprocess" / "docred"
        self.train_path = self.data_dir / docred.TRAIN_FILENAME
        self.valid_path = self.data_dir / docred.VALID_FILENAME
        self.test_path = self.data_dir / docred.TEST_FILENAME
        # This is the dictionary mapping rel ids to rel labels provided by DocRED
        types = json.loads((self.data_dir / docred.TYPES_FILENAME).read_text())
        self.rel_labels = {key: value["verbose"] for key, value in types["relations"].items()}

        # The expected data, after preprocessing, for each partition
        self.train = [
            (
                "Gambier Island is an island located in Howe Sound near Vancouver , British Columbia"
                " . It is about in size and is located about north of the Horseshoe Bay community"
                " and ferry terminal in westernmost West Vancouver . A rugged and sparsely populated"
                " island , it is far quieter than its neighbour Bowen Island , which is popular with"
                " day - trippers and summer vacationers . Gambier Island is accessible only by B.C."
                " Ferries passenger service , water taxi or other boats . There is no central road"
                " network . The island elects two trustees to the Islands Trust , an organization"
                " that unites small island communities in British Columbia to oversee development"
                " and land use . Other islands in Howe Sound include Keats Island and Anvil Island ."
                "\tkeats island @LOC@ howe sound @LOC@ @LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY@"
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
        # The second example here contains the special case where a relation has the same entities,
        # but a different relation type. This tests that we sort that accordingly, i.e. first
        # lexicographically and then by order of first appearance.
        self.valid = [
            (
                "David Thomas McLaughlin ( March 16 , 1932 – August 25 , 2004 ) was the 14th"
                " President of Dartmouth College , 1981 – 1987 . McLaughlin also served as chief"
                " executive officer of Orion Safety Products from 1988 to December 31 , 2000 . He"
                " was president and chief executive officer of the Aspen Institute from 1988 to 1997"
                " and its chairman from 1987 to 1988 . He served as chairman and chief executive"
                " officer of Toro Company from 1977 to 1981 , after serving in various management"
                " positions at Toro Company since 1970 . McLaughlin served as a director of CBS"
                " Corporation from 1979 , becoming chairman of the board in January 1999 until the"
                " CBS merger . He also served as a director of Infininity Broadcasting Corporation"
                " until the Infinity merger ."
                "\tdavid thomas mclaughlin ; mclaughlin @PER@ march 16 , 1932 @TIME@ @DATE_OF_BIRTH@"
                " david thomas mclaughlin ; mclaughlin @PER@ august 25 , 2004 @TIME@ @DATE_OF_DEATH@"
                " david thomas mclaughlin ; mclaughlin @PER@ cbs corporation ; cbs @ORG@ @EMPLOYER@"
            ),
            (
                "Chachalacas are galliform birds from the genus Ortalis . These birds are found in"
                " wooded habitats in far southern United States ( Texas ) , Mexico , and Central and"
                " South America . They are social , can be very noisy and often remain fairly common"
                " even near humans , as their relatively small size makes them less desirable to"
                " hunters than their larger relatives . As invasive pests , they have a ravenous"
                " appetite for tomatoes , melons , beans , and radishes and can ravage a small garden"
                " in short order . They travel in packs of six to twelve . They somewhat resemble the"
                " guans , and the two have commonly been placed in a subfamily together , though the"
                " chachalacas are probably closer to the curassows . The generic name is derived"
                ' from the Greek word όρταλις , meaning " pullet " or " domestic hen . " The common'
                " name is an onomatopoeia for the four - noted cackle of the plain chachalaca"
                " ( O. vetula ) . Mitochondrial and nuclear DNA sequence data tentatively suggest"
                " that the chachalacas emerged as a distinct lineage during the Oligocene , somewhere"
                " around 40–20 mya , possibly being the first lineage of modern cracids to evolve ;"
                " this does agree with the known fossil record – including indeterminate , cracid -"
                " like birds – which very cautiously favors a north - to - south expansion of the"
                " family .\t"
                "chachalacas @MISC@ ortalis @MISC@ @PARENT_TAXON@"
                " united states @LOC@ texas @LOC@ @CONTAINS_ADMINISTRATIVE_TERRITORIAL_ENTITY@"
                " texas @LOC@ united states @LOC@ @LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY@"
                " texas @LOC@ united states @LOC@ @COUNTRY@"
            ),
        ]
        self.test = [
            (
                "The Battle class were a class of destroyers of the British Royal Navy ( RN ) and"
                " Royal Australian Navy ( RAN ) , named after naval or other battles fought by"
                " British or English forces . Built in three groups , the first group were ordered"
                " under the 1942 naval estimates . A modified second and third group , together with"
                " two ships of an extended design were planned for the 1943 and 1944 estimates ."
                " Most of these ships were cancelled when it became apparent that the war was being"
                " won and the ships would not be required , although two ships of the third group ,"
                " ordered for the RAN , were not cancelled and were subsequently completed in"
                " Australia . Seven Battles were commissioned before the end of World War II , but"
                " only saw action , with the British Pacific Fleet .\t"
                "british royal navy ; rn @ORG@ british @LOC@ @COUNTRY@"
                " british royal navy ; rn @ORG@ english @LOC@ @COUNTRY@"
                " royal australian navy ; ran @ORG@ australia @LOC@ @COUNTRY@"
                " royal australian navy ; ran @ORG@ world war ii @MISC@ @CONFLICT@"
                " british pacific fleet @ORG@ british @LOC@ @COUNTRY@"
                " british pacific fleet @ORG@ english @LOC@ @COUNTRY@"
                " british pacific fleet @ORG@ world war ii @MISC@ @CONFLICT@"
            )
        ]

    def test_preprocess_docred(self) -> None:
        # training data
        train_raw = json.loads(Path(self.train_path).read_text())
        actual = docred._preprocess(train_raw, rel_labels=self.rel_labels)
        assert actual == self.train

        # validation data
        valid_raw = json.loads(Path(self.valid_path).read_text())
        actual = docred._preprocess(valid_raw, rel_labels=self.rel_labels)
        assert actual == self.valid

        # test data
        test_raw = json.loads(Path(self.test_path).read_text())
        actual = docred._preprocess(test_raw, rel_labels=self.rel_labels)
        assert actual == self.test

    def test_docred_command(self, tmp_path: Path) -> None:
        output_dir = str(tmp_path)
        result = runner.invoke(docred.app, [output_dir])
        assert result.exit_code == 0

        # Check that the expected files were created
        assert (tmp_path / "train.tsv").is_file()
        assert (tmp_path / "valid.tsv").is_file()
        assert (tmp_path / "test.tsv").is_file()
