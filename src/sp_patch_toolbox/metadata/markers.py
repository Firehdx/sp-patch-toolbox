import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _ascii_cleanup(text: str) -> str:
    replacements = {
        "\u03b1": "a",
        "\u03b2": "beta",
        "\u03b3": "gamma",
        "\u03b4": "delta",
        "\uff1a": ":",
        "\u2011": "-",
        "\u2013": "-",
        "\u2014": "-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def normalize_marker_text(raw_name: Optional[str]) -> str:
    if raw_name is None:
        return ""
    text = _ascii_cleanup(str(raw_name)).strip()
    text = re.sub(r"\s+", " ", text)
    if re.match(r"^(empty|blank|background)\b", text, flags=re.IGNORECASE):
        return "blank"
    text = re.sub(r"^Panel\s*\d+\s*:?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^Panel\s*:?\s*", "", text, flags=re.IGNORECASE)
    if ":" in text:
        text = text.split(":")[-1].strip()
    text = re.sub(r"^(Hoechst|HOECHST)\s*\d+$", "Hoechst", text, flags=re.IGNORECASE)
    if " - " in text:
        head = text.split(" - ", 1)[0].strip()
        if re.search(r"[A-Za-z0-9]", head):
            text = head
    text = re.sub(r"^Cycle\s*\d+[_\s-]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\d+\s*[_-]\s*", "", text)
    # A number after a slash is a panel-position suffix in several HTAN
    # exports (for example ``AGR2/016``), rather than part of the marker.
    text = re.sub(r"/\d{3}$", "", text)
    text = re.sub(r"-+$", "", text).strip()
    # Normalize common cyclic-panel technical planes before registry lookup.
    # They must not create separate biological-marker identities.
    if re.fullmatch(r"(?:DAPI|DNA)[\s_-]*\d+", text, flags=re.IGNORECASE):
        return "DAPI"
    if re.fullmatch(r"Hoech(?:st|est)[\s_-]*\d+", text, flags=re.IGNORECASE):
        return "Hoechst"
    if text.upper() == "HEM":
        return "Nuclei"
    if re.fullmatch(r"(?:blank|empty)\s*\d*[A-Za-z]*", text, flags=re.IGNORECASE):
        return "blank"
    if re.fullmatch(r"(?:control[-\s]*\d+nm|A(?:488|555|647)|anti[-\s]*(?:goat|mouse|rabbit))", text, flags=re.IGNORECASE):
        return "blank"
    if re.fullmatch(r"C\d+", text, flags=re.IGNORECASE) or re.fullmatch(
        r"cycle[_\s]*\d+[_\s]*channel[_\s]*\d+", text, flags=re.IGNORECASE
    ):
        return "unmapped"
    if re.fullmatch(r"AF\d+[A-Za-z]*", text, flags=re.IGNORECASE):
        return "blank"
    text = re.sub(r"[_\s-]*Argo(?:Fluor)?\d+[A-Za-z]*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[_\s-]*(?:AF|DL|BL)\d+[A-Za-z]*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[_\s-]*(?:FITC|PE|APC|Cy\d+)$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text)
    text = text.replace("_", " ").strip()
    text = re.sub(r"-+$", "", text).strip()
    text = re.sub(r"\s+", " ", text)
    # Cross-platform spelling and clone aliases. Values either match an
    # existing registry marker or a stable entry in ``auto_markers``.
    alias_key = re.sub(r"[^a-z0-9]+", "", text.lower())
    aliases = {
        "androgenreceptor": "AR",
        "antigenki67": "Ki67",
        "aorticsmoothmuscleactin": "aSMA",
        "alphasma": "aSMA",
        "cd20ms4a1": "CD20",
        "ms4a1": "CD20",
        "cd279": "PD-1",
        "cd3d": "CD3",
        "cd8r": "CD8",
        "cdh1": "E-Cadherin",
        "ecad": "E-Cadherin",
        "ck14": "Keratin14",
        "ck5": "Keratin5",
        "ck17": "Keratin 17",
        "ck19": "Keratin 19",
        "ck8": "Keratin8",
        "krt18": "Keratin 18",
        "ck18": "Keratin 18",
        "coliv": "Collagen IV",
        "cox2ptgs2": "COX2",
        "epidermalgrowthfactorreceptor": "EGFR",
        "grnzb": "GZMB",
        "grzb": "GZMB",
        "grzb001": "GZMB",
        "granb": "GZMB",
        "h2ax": "H2AX",
        "gh2ax": "H2AX",
        "histoneh2ax": "H2AX",
        "hif1a": "HIF1A",
        "her2": "HER2",
        "ido": "IDO1",
        "krt": "Cytokeratin",
        "lamac": "LaminA",
        "laminac": "LaminA",
        "laminabc": "LaminA",
        "mki67": "Ki67",
        "mlna": "Melan A",
        "nestin": "Nestin",
        "olig1": "OLIG1",
        "p16": "P16",
        "p21": "P21",
        "pdpn": "Podoplanin",
        "pancytokrt": "Cytokeratin",
        "pankrt": "Cytokeratin",
        "synaphysin": "Synaptophysin",
        "tmem173": "STING",
        "vim": "Vimentin",
        "vimentine": "Vimentin",
        "ckit": "CD117",
        "pstat3": "pSTAT3",
        "prnapoliictd": "RNA polymerase II CTD",
    }
    return aliases.get(alias_key, text)


def marker_key(raw_name: Optional[str]) -> str:
    text = normalize_marker_text(raw_name).lower()
    return re.sub(r"[^a-z0-9]+", "", text)


class MarkerRegistry:
    """Stable raw channel name to marker id mapping.

    The JSON registry is deliberately easy to edit. Unknown channels remain
    usable for pretraining and are emitted as id 0.
    """

    def __init__(self, payload: Dict):
        self.payload = payload
        self.unknown_marker_id = int(payload.get("unknown_marker_id", 0))
        self.blank_marker_id = int(payload.get("blank_marker_id", 1))
        self._id_to_name: Dict[int, str] = {}
        self._key_to_id: Dict[str, int] = {}
        for item in payload.get("markers", []):
            marker_id = int(item["id"])
            name = str(item["name"])
            self._id_to_name[marker_id] = name
            for alias in [name] + list(item.get("aliases", [])):
                key = marker_key(alias)
                if key:
                    self._key_to_id[key] = marker_id
        # ``auto_markers`` is a reviewed, append-only extension generated
        # from marker labels already present in the patch manifests.  Keeping
        # it separate from the hand-curated synonym table makes expansion
        # auditable while preserving every existing numeric ID.
        next_marker_id = max(self._id_to_name, default=-1) + 1
        for value in payload.get("auto_markers", []):
            if isinstance(value, dict):
                name = str(value["name"])
                aliases = list(value.get("aliases", []))
            else:
                name = str(value)
                aliases = []
            key = marker_key(name)
            if not key or key in self._key_to_id:
                continue
            marker_id = next_marker_id
            next_marker_id += 1
            self._id_to_name[marker_id] = name
            for alias in [name] + aliases:
                alias_key = marker_key(alias)
                if alias_key:
                    self._key_to_id[alias_key] = marker_id

    @classmethod
    def from_json(cls, path: str | Path) -> "MarkerRegistry":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls(json.load(handle))

    def id_for(self, raw_name: Optional[str]) -> int:
        key = marker_key(raw_name)
        if not key:
            return self.unknown_marker_id
        if key in self._key_to_id:
            return self._key_to_id[key]
        compact = re.sub(r"[^A-Za-z0-9/-]+", "", normalize_marker_text(raw_name))
        return self._key_to_id.get(marker_key(compact), self.unknown_marker_id)

    def ids_for(self, raw_names: Iterable[Optional[str]]) -> List[int]:
        return [self.id_for(name) for name in raw_names]

    def name_for_id(self, marker_id: int) -> str:
        return self._id_to_name.get(int(marker_id), "Unknown")

    def names_for_ids(self, marker_ids: Iterable[int]) -> List[str]:
        return [self.name_for_id(marker_id) for marker_id in marker_ids]
