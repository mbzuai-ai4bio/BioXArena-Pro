"""Check training/peptide_v2_format.py against the official grader (run outside the sandbox, from the Code root)."""
import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import tasks.native_manifest_metrics as N
import peptide_v2_format as V
assert V._PEPTIDE_V2_RESIDUE_MASS == N._PEPTIDE_V2_RESIDUE_MASS and V._PEPTIDE_V2_MOD_ALIASES == N._PEPTIDE_V2_MOD_ALIASES
for s in ["PEPTIDE", "M(ox)PEPC[+57.021]K", "(+42.011)AKM[15.9949]", "Q(pyro-glu)K", "", "pepTIDE", "AX", "A(", "A(foo)", "nan", "A" * 61, "(ac)"]:
    def run(f):
        try: return ("ok", f(s))
        except Exception as e: return ("err", type(e).__name__ != "")
    a, b = run(N._peptide_v2_masses), run(V._peptide_v2_masses)
    assert a == b, (s, a, b)
print("peptide_v2_format matches grader")
