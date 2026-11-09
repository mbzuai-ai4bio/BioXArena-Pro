"""Peptide notation parser of the official de-novo grader (tasks/native_manifest_metrics.py, _peptide_v2_masses),
vendored so the submission preflight can run inside the agent sandbox, which cannot read tasks/. Generated 2026-10-19 by
copying the grader's constants and functions; peptide_v2_format_test() checks it still matches the grader."""
import math, re
from typing import Any

_PEPTIDE_V2_CARBAMIDOMETHYL = 57.02146
_PEPTIDE_V2_RESIDUE_MASS = {'A': 71.03711, 'R': 156.10111, 'N': 114.04293, 'D': 115.02694, 'C': 160.03065, 'E': 129.04259, 'Q': 128.05858, 'G': 57.02146, 'H': 137.05891, 'I': 113.08406, 'L': 113.08406, 'K': 128.09496, 'M': 131.04049, 'F': 147.06841, 'P': 97.05276, 'S': 87.03203, 'T': 101.04768, 'W': 186.07931, 'Y': 163.06333, 'V': 99.06841}
_PEPTIDE_V2_MOD_ALIASES = {'ox': 15.9949, 'oxidation': 15.9949, 'cam': 57.02146, 'carbamidomethyl': 57.02146, 'carbamidomethylation': 57.02146, 'deamidation': 0.984, 'deamidated': 0.984, 'acetyl': 42.0106, 'acetylation': 42.0106, 'ac': 42.0106, 'carbamyl': 43.0058, 'carbamylation': 43.0058, 'ammonia-loss': -17.0265, 'ammonia_loss': -17.0265, 'nh3-loss': -17.0265, 'nh3 loss': -17.0265, 'pyro-glu': -17.0265, 'gln->pyro-glu': -17.0265, 'glu->pyro-glu': -18.0106, 'deam': 0.984, 'oxidised': 15.9949, 'oxidized': 15.9949}
_PEPTIDE_V2_MAX_RESIDUES = 60


def _peptide_v2_mod_delta(text: str) -> float:
    value = text.strip().lower()
    if value.endswith("da"):
        value = value[:-2].strip()
    if value in _PEPTIDE_V2_MOD_ALIASES:
        return _PEPTIDE_V2_MOD_ALIASES[value]
    match = re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", value)
    if not match:
        raise ValueError(f"Unsupported peptide modification: {text!r}")
    return float(match.group(0))


def _peptide_v2_masses(sequence: Any, max_residues: int | None = _PEPTIDE_V2_MAX_RESIDUES) -> list[float]:
    """Residue mass list of an annotated peptide string (empty list for an empty prediction)."""
    if sequence is None or (isinstance(sequence, float) and math.isnan(sequence)):
        return []
    text = str(sequence).strip()
    masses: list[float] = []
    pending_nterm = 0.0
    index = 0
    while index < len(text):
        char = text[index]
        if char in ".-_ ":
            index += 1
            continue
        if char in "([":
            # annotation before the first residue: N-terminal modification
            if masses:
                raise ValueError(f"Dangling modification in peptide {text!r}")
            closing = ")" if char == "(" else "]"
            end = text.find(closing, index + 1)
            if end < 0:
                raise ValueError("Unclosed peptide modification")
            pending_nterm += _peptide_v2_mod_delta(text[index + 1:end])
            index = end + 1
            continue
        residue = char  # upper-case one-letter codes only; a literal "nan" is rejected
        if residue not in _PEPTIDE_V2_RESIDUE_MASS:
            raise ValueError(f"Unsupported peptide residue: {char!r}")
        mass = _PEPTIDE_V2_RESIDUE_MASS[residue]
        index += 1
        while index < len(text) and text[index] in "([":
            opening = text[index]
            closing = ")" if opening == "(" else "]"
            end = text.find(closing, index + 1)
            if end < 0:
                raise ValueError("Unclosed peptide modification")
            delta = _peptide_v2_mod_delta(text[index + 1:end])
            if residue == "C" and abs(delta - _PEPTIDE_V2_CARBAMIDOMETHYL) < 0.05:
                delta = 0.0  # explicit carbamidomethyl on C: already fixed
            mass += delta
            index = end + 1
        if not masses:
            mass += pending_nterm
            pending_nterm = 0.0
        masses.append(mass)
    if pending_nterm and not masses:
        raise ValueError(f"Modification without residue in peptide {text!r}")
    if max_residues is not None and len(masses) > max_residues:
        raise ValueError(
            f"Peptide prediction longer than {_PEPTIDE_V2_MAX_RESIDUES} residues"
        )
    return masses
