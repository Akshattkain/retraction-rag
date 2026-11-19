"""
Retraction reason -> severity class map (no network).

Classes, highest first: A > B > C > D. Papers with no reasons are "U".
Views: strict = A; broad = A or B; all = every retracted paper.

Importing this module asserts the class sizes (31 + 46 + 15 + 18 = 110) and that no reason
appears in two classes. check_against_parquet() asserts that every distinct reason in
data/processed/retracted.parquet is mapped and that the map has no extra keys.
"""

from pathlib import Path

import pandas as pd

RETRACTED_PATH = Path("data/processed/retracted.parquet")

_A = """Unreliable Results and/or Conclusions; Concerns/Issues about Data; Paper Mill; Concerns/Issues about Results and/or Conclusions; Duplication of/in Image; Computer-Aided Content or Computer-Generated Content; Concerns/Issues about Image; Original Data and/or Images not Provided and/or not Available; Unreliable Data; Error in Data; Falsification/Fabrication of Data; Manipulation of Images; Error in Analyses; Error in Results and/or Conclusions; Error in Methods; Error in Image; Results Not Reproducible; Falsification/Fabrication of Image; Error in Text; Unreliable Image; Concerns/Issues about Methods; Cites Retracted Work; Contamination of Cell Lines/Tissues; Error in Materials; Error in Cell Lines/Tissues; Falsification/Fabrication of Results; Contamination of Materials; Manipulation of Results; Manipulation of Data; Hoax Paper; Sabotage of Materials/Methods"""

_B = """Concerns/Issues about Referencing/Attributions; Concerns/Issues about Peer Review; Compromised Peer Review; Objections by Author(s); Lack of IRB/IACUC Approval and/or Compliance; Concerns/Issues about Authorship/Affiliation; Euphemisms for Duplication; Misconduct by Author; Euphemisms for Plagiarism; Misconduct - Official Investigation(s) and/or Finding(s); Duplication of/in Article; Informed/Patient Consent - None/Withdrawn; Plagiarism of/in Article; Duplication of Data; Plagiarism of Image; Objections by Third Party; Conflict of Interest; Concerns/Issues about Human Subject Welfare; False/Forged Authorship; Plagiarism of Text; Plagiarism of Data; Ethical Violations by Author; Rogue Editor; Concerns/Issues about Third Party Involvement; Lack of Approval from Author; Copyright Claims; Breach of Policy by Author; Lack of Approval from Third Party; Duplication of Text; Lack of Approval from Company/Institution; False/Forged Affiliation; Taken from Dissertation/Thesis; Criminal Proceedings; Legal Reasons and/or Threats; Bias Issues or Lack of Balance; Concerns/Issues about Animal Welfare; Publishing Ban; Civil Proceedings; Ethical Violations by Company/Institution/Third Party; Objections by Company/Institution; Misconduct by Third Party; Salami Slicing; Euphemisms for Misconduct; Taken via Peer Review; Misconduct by Company/Institution; Taken via Translation"""

_C = """Investigation by Journal/Publisher; Investigation by Third Party; Investigation by Company/Institution; Investigation by ORI; Author Unresponsive; Concerns/Issues about Article; Miscommunication with/by Author; Complaints about Author; Error by Third Party; Miscommunication with/by Journal/Publisher; Miscommunication with/by Third Party; Miscommunication with/by Company/Institution; Complaints about Third Party; Complaints about Company/Institution; No Further Action"""

_D = """Upgrade/Update of Prior Notice(s); Date of Article and/or Notice Unknown; Notice - Limited or No Information; Removed; Retract and Replace; Duplication of Content through Error by Journal/Publisher; Error by Journal/Publisher; Withdrawn as Out of Date; Notice - Unable to Access via current resources; Notice - Lack of; Doing the Right Thing; Withdrawn to Publish in Different Journal; Temporary Removal; Updated to Retraction; Nonpayment of Fees and/or Refusal to Pay; Not Presented at Conference; Updated to Correction; Transfer of Copyright and/or Ownership"""

CLASS_SIZES = {"A": 31, "B": 46, "C": 15, "D": 18}

REASON_CLASS = {}
for _cls, _text in [("A", _A), ("B", _B), ("C", _C), ("D", _D)]:
    _reasons = [r.strip() for r in _text.split(";")]
    assert len(_reasons) == CLASS_SIZES[_cls], (_cls, len(_reasons))
    assert len(set(_reasons)) == len(_reasons), f"duplicate reason inside class {_cls}"
    for _r in _reasons:
        assert _r not in REASON_CLASS, f"reason in two classes: {_r!r}"
        REASON_CLASS[_r] = _cls
assert len(REASON_CLASS) == 110

# Higher number = more severe. "U" (no reasons) is below D.
CLASS_RANK = {"U": 0, "D": 1, "C": 2, "B": 3, "A": 4}
RANK_CLASS = {v: k for k, v in CLASS_RANK.items()}
CLASSES = ["A", "B", "C", "D", "U"]


def paper_class(reasons):
    """Highest class among a paper's reasons; "U" if it has none."""
    if reasons is None or len(reasons) == 0:
        return "U"
    return RANK_CLASS[max(CLASS_RANK[REASON_CLASS[r]] for r in reasons)]


def check_against_parquet(path=RETRACTED_PATH):
    """Every distinct reason in the parquet is mapped, and the map has no extra keys."""
    df = pd.read_parquet(path, columns=["reasons"])
    seen = set()
    for reasons in df["reasons"]:
        if reasons is not None:
            seen.update(reasons)
    unmapped = seen - set(REASON_CLASS)
    extra = set(REASON_CLASS) - seen
    assert not unmapped, f"unmapped reasons: {sorted(unmapped)}"
    assert not extra, f"map keys not in data: {sorted(extra)}"
    assert len(seen) == len(REASON_CLASS) == 110
    return len(seen)


if __name__ == "__main__":
    n = check_against_parquet()
    print(f"OK: {n} distinct reasons, all mapped, no extra keys "
          f"({' + '.join(str(CLASS_SIZES[c]) for c in 'ABCD')})")
