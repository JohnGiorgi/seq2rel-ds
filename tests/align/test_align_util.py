from seq2rel_ds.align import util


def test_get_uniprot_synonyms_one_id() -> None:
    """Test that the function works for a single Uniprot ID component names."""
    expected = {
        "Q09670": ["Putative NADPH dehydrogenase C5H10.04", "Old yellow enzyme homolog 1"],
    }

    actual = util.get_uniprot_synonyms(uniprot_ids=list(expected.keys()))
    assert actual == expected


def test_get_uniprot_synonyms_gene_names() -> None:
    """Test that the function captures gene names."""
    expected = {
        "Q86V15": [
            "Zinc finger protein castor homolog 1",
            "Castor-related protein",
            "Putative survival-related protein",
            "Zinc finger protein 693",
            "CASZ1",
            "CST",
            "SRG",
            "ZNF693",
        ],
    }

    actual = util.get_uniprot_synonyms(uniprot_ids=list(expected.keys()))
    assert actual == expected


def test_get_uniprot_synonyms_ignores_ids() -> None:
    """Test that any IDs provided as part of the name are ignored."""
    expected = {
        "Q20819": ["Elongation factor Ts, mitochondrial", "EF-Ts", "EF-TsMt", "tsfm-1"],
    }
    actual = util.get_uniprot_synonyms(uniprot_ids=list(expected.keys()))
    assert actual == expected


def test_get_uniprot_synonyms_component_names() -> None:
    """Test that the function captures component names."""
    expected = {
        "P33587": [
            "Vitamin K-dependent protein C",
            "Anticoagulant protein C",
            "Autoprothrombin IIA",
            "Blood coagulation factor XIV",
            "Vitamin K-dependent protein C light chain",
            "Vitamin K-dependent protein C heavy chain",
            "Activation peptide",
            "Proc",
        ],
    }

    actual = util.get_uniprot_synonyms(uniprot_ids=list(expected.keys()))
    assert actual == expected


def test_get_uniprot_synonyms_many_ids() -> None:
    expected = {
        "Q09670": ["Putative NADPH dehydrogenase C5H10.04", "Old yellow enzyme homolog 1"],
        "Q86V15": [
            "Zinc finger protein castor homolog 1",
            "Castor-related protein",
            "Putative survival-related protein",
            "Zinc finger protein 693",
            "CASZ1",
            "CST",
            "SRG",
            "ZNF693",
        ],
        "Q20819": ["Elongation factor Ts, mitochondrial", "EF-Ts", "EF-TsMt", "tsfm-1"],
        "P33587": [
            "Vitamin K-dependent protein C",
            "Anticoagulant protein C",
            "Autoprothrombin IIA",
            "Blood coagulation factor XIV",
            "Vitamin K-dependent protein C light chain",
            "Vitamin K-dependent protein C heavy chain",
            "Activation peptide",
            "Proc",
        ],
    }

    actual = util.get_uniprot_synonyms(uniprot_ids=list(expected.keys()))
    assert actual == expected


def test_get_pubtator_response() -> None:
    pmid = "28483577"
    expected = {
        pmid: {
            "title": {
                "text": (
                    "Formoterol and fluticasone propionate combination improves histone"
                    " deacetylation and anti-inflammatory activities in bronchial epithelial cells"
                    " exposed to cigarette smoke."
                ),
                "ents": [],
            },
            "abstract": {
                "text": (
                    "BACKGROUND: The addition of long-acting beta2-agonists (LABAs) to"
                    " corticosteroids improves asthma control. Cigarette smoke exposure, increasing"
                    " oxidative stress, may negatively affect corticosteroid responses."
                    " The anti-inflammatory effects of formoterol (FO) and fluticasone propionate (FP)"
                    " in human bronchial epithelial cells exposed to cigarette smoke extracts (CSE) are"
                    " unknown. AIMS: This study explored whether FP, alone and in combination with FO,"
                    " in human bronchial epithelial cellline (16-HBE) and primary bronchial epithelial"
                    " cells (NHBE), counteracted some CSE-mediated effects and in particular some of the"
                    " molecular mechanisms of corticosteroid resistance. METHODS: 16-HBE and NHBE were"
                    " stimulated with CSE, FP and FO alone or combined. HDAC3 and HDAC2 activity, nuclear"
                    " translocation of GR and NF-kappaB, pERK1/2/tERK1/2 ratio, IL-8, TNF-alpha,"
                    " IL-1beta mRNA expression, and mitochondrial ROS were evaluated. Actin reorganization"
                    " in neutrophils was assessed by fluorescence microscopy using the phalloidin method."
                    " RESULTS: In 16-HBE, CSE decreased expression/activity of HDAC3, activity of HDAC2,"
                    " nuclear translocation of GR and increased nuclear NF-kappaB expression, pERK 1/2/tERK1/2"
                    " ratio, and mRNA expression of inflammatory cytokines. In NHBE, CSE increased mRNA"
                    " expression of inflammatory cytokines and supernatants from CSE exposed NHBE increased"
                    " actin reorganization in neutrophils. FP combined with FO reverted all these phenomena"
                    " in CSE stimulated 16-HBE cells as well as in NHBE cells. CONCLUSIONS: The present"
                    " study provides compelling evidences that FP combined with FO may contribute to revert"
                    " some processes related to steroid resistance induced by oxidative stress due to cigarette"
                    " smoke exposure increasing the anti-inflammatory effects of FP."
                ),
                "ents": [
                    "HDAC3",
                    "HDAC2",
                    "NF-kappaB",
                    "IL-8",
                    "TNF-alpha",
                    "IL-1beta",
                    "HDAC3",
                    "HDAC2",
                    "NF-kappaB",
                ],
            },
        }
    }

    actual = util.get_pubtator_response(pmids=[pmid], concepts=["gene"])
    assert actual == expected
