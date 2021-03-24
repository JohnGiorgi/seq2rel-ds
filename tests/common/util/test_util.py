from seq2rel_ds.common.util import get_pubtator_response


def test_get_pubtator_response() -> None:
    pmid = "28483577"
    expected = {
        pmid: {
            "title": {
                "text": "Formoterol and fluticasone propionate combination improves histone deacetylation and anti-inflammatory activities in bronchial epithelial cells exposed to cigarette smoke.",
                "ents": [],
            },
            "abstract": {
                "text": "BACKGROUND: The addition of long-acting beta2-agonists (LABAs) to corticosteroids improves asthma control. Cigarette smoke exposure, increasing oxidative stress, may negatively affect corticosteroid responses. The anti-inflammatory effects of formoterol (FO) and fluticasone propionate (FP) in human bronchial epithelial cells exposed to cigarette smoke extracts (CSE) are unknown. AIMS: This study explored whether FP, alone and in combination with FO, in human bronchial epithelial cellline (16-HBE) and primary bronchial epithelial cells (NHBE), counteracted some CSE-mediated effects and in particular some of the molecular mechanisms of corticosteroid resistance. METHODS: 16-HBE and NHBE were stimulated with CSE, FP and FO alone or combined. HDAC3 and HDAC2 activity, nuclear translocation of GR and NF-kappaB, pERK1/2/tERK1/2 ratio, IL-8, TNF-alpha, IL-1beta mRNA expression, and mitochondrial ROS were evaluated. Actin reorganization in neutrophils was assessed by fluorescence microscopy using the phalloidin method. RESULTS: In 16-HBE, CSE decreased expression/activity of HDAC3, activity of HDAC2, nuclear translocation of GR and increased nuclear NF-kappaB expression, pERK 1/2/tERK1/2 ratio, and mRNA expression of inflammatory cytokines. In NHBE, CSE increased mRNA expression of inflammatory cytokines and supernatants from CSE exposed NHBE increased actin reorganization in neutrophils. FP combined with FO reverted all these phenomena in CSE stimulated 16-HBE cells as well as in NHBE cells. CONCLUSIONS: The present study provides compelling evidences that FP combined with FO may contribute to revert some processes related to steroid resistance induced by oxidative stress due to cigarette smoke exposure increasing the anti-inflammatory effects of FP.",
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

    actual = get_pubtator_response(pmids=[pmid], concepts=["gene"])
    assert actual == expected
