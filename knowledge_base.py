"""
knowledge_base.py
-----------------
Loads the symbolic IPC knowledge base (data/ipc_kb.json, built by kb/build_kb.py)
and exposes the legal structure the reasoner needs:

- section lookup and id normalisation ("u/s 302 IPC" -> "302")
- definition -> punishing section mapping (s.378 theft is punished under s.379)
- the lesser-included offence hierarchy (s.392 robbery contains s.379 theft)
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

KB_PATH = Path(__file__).parent / "data" / "ipc_kb.json"

# Curated greater -> lesser-included offences. Merged with the LLM-drafted
# `lesser_included` lists in the KB. If a greater offence is established, its
# lesser offences are not charged separately; if a greater offence fails on one
# ingredient, the court may convict for a lesser offence whose ingredients hold.
CURATED_LESSER = {
    "302": ["304", "304A", "307", "308", "326", "325", "324", "323"],
    "300": ["304", "304A", "307", "308", "326", "325", "324", "323"],
    "304": ["304A", "308", "326", "325", "324", "323"],
    "299": ["304A", "308", "326", "325", "324", "323"],
    "304B": ["306"],  # 498A is a distinct offence and is charged alongside 304B
    "307": ["308", "326", "325", "324", "323", "506"],
    "308": ["325", "324", "323"],
    "396": ["395", "391", "392", "390", "397", "394", "379", "378", "302", "300"],
    "395": ["392", "390", "394", "379", "378", "384", "383"],
    "391": ["392", "390", "394", "379", "378", "384", "383"],
    "397": ["392", "390", "394", "379", "378", "323"],
    "394": ["392", "390", "379", "378", "323"],
    "392": ["379", "378", "384", "383", "356"],
    "390": ["379", "378", "384", "383", "356"],
    "382": ["380", "379", "378"],
    "380": ["379", "378"],
    "381": ["379", "378"],
    "386": ["384", "383", "506", "503"],
    "412": ["411"],
    "409": ["406", "405", "403"],
    "408": ["406", "405", "403"],
    "407": ["406", "405", "403"],
    "406": ["403"],
    "405": ["403"],
    "420": ["417", "415"],
    "418": ["417", "415"],
    "419": ["417", "415"],
    "467": ["465", "463"],
    "468": ["465", "463"],
    "469": ["465", "463"],
    "458": ["457", "454", "451", "448", "442", "447", "441", "445"],
    "457": ["454", "451", "448", "442", "447", "441", "445"],
    "454": ["451", "448", "442", "447", "441", "445"],
    "451": ["448", "442", "447", "441"],
    "448": ["447", "441"],
    "442": ["447", "441"],
    "436": ["435", "426", "425"],
    "435": ["426", "425"],
    "326A": ["326B", "326", "325", "324", "323", "320", "319"],
    "326": ["325", "324", "323", "320", "319"],
    "325": ["323", "319"],
    "320": ["323", "319"],
    "324": ["323", "319"],
    "332": ["323", "319", "353", "352", "351"],
    "353": ["352", "351", "350", "186"],
    "376D": ["376", "375", "354", "354A"],
    "376A": ["376", "375", "354"],
    "376": ["354", "354A", "354B", "509"],
    "375": ["354", "354A", "354B", "509"],
    "376C": ["354", "354A"],
    "376B": ["354"],
    "354B": ["354", "354A", "509", "352"],
    "354": ["354A", "509", "352", "351"],
    "354A": ["509"],
    "354D": ["509"],
    "364A": ["364", "365", "363", "359", "361", "362", "342", "506"],
    "364": ["365", "363", "359", "361", "362"],
    "366": ["365", "363", "359", "361", "362"],
    "367": ["365", "363", "359", "361", "362"],
    "369": ["363", "359", "361", "362"],
    "363A": ["363", "359", "361"],
    "365": ["363", "359", "361", "362", "342", "340"],
    "346": ["343", "342", "340", "341", "339"],
    "343": ["342", "340", "341", "339"],
    "342": ["341", "339"],
    "340": ["341", "339"],
    "148": ["147", "146", "144", "143", "141"],
    "147": ["143", "141"],
    "146": ["143", "141"],
    "144": ["143", "141"],
    "295A": ["298"],
    "507": ["506", "503"],
    "489A": ["489C"],
    "232": ["231"],
    "255": ["259"],
    "306": [],
}

_SECTION_REF = re.compile(
    r"(?:\bsection|\bsec\.?|\bs\.|\bu/s\.?)\s*(\d{2,3}[a-z]?)\b"
    r"|\b(\d{2,3}[a-z]?)\s*(?:of\s+the\s+)?(?:ipc\b|i\.p\.c|indian\s+penal\s+code)",
    re.IGNORECASE,
)


class KnowledgeBase:
    def __init__(self, path: Path = KB_PATH):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Build it with: python -m kb.build_kb")
        self.sections: Dict[str, dict] = json.loads(path.read_text(encoding="utf-8"))
        self._lesser = {}
        for sec, entry in self.sections.items():
            merged = CURATED_LESSER.get(sec, []) + entry.get("lesser_included", [])
            self._lesser[sec] = [s for s in dict.fromkeys(merged) if s in self.sections and s != sec]

    def __contains__(self, sec: str) -> bool:
        return sec in self.sections

    def get(self, sec: str) -> Optional[dict]:
        return self.sections.get(self.normalize(sec) or "")

    def normalize(self, raw) -> Optional[str]:
        """'IPC_302', 'S. 302', '302 IPC', '498a' -> canonical KB id, or None."""
        if raw is None:
            return None
        m = re.search(r"(\d{2,3})\s*([A-Z]{0,2})\b", str(raw).upper().replace("_", " "))
        if not m:
            return None
        for s in (m.group(1) + m.group(2), m.group(1)):
            if s in self.sections:
                return s
        return None

    def charge_section(self, sec: str) -> str:
        """Section under which the offence is punished (definition sections map to it)."""
        under = self.sections[sec]["punishment"].get("punishable_under")
        return under if under in self.sections else sec

    def same_offence(self, a: str, b: str) -> bool:
        return a == b or self.charge_section(a) == self.charge_section(b)

    def lesser_of(self, sec: str) -> List[str]:
        return self._lesser.get(sec, [])

    def is_lesser(self, lesser: str, greater: str) -> bool:
        if self.same_offence(lesser, greater):
            return False
        lesser_c = self.charge_section(lesser)
        for g in {greater, self.charge_section(greater)}:
            for l in self.lesser_of(g):
                if l == lesser or self.charge_section(l) == lesser_c:
                    return True
        return False

    def find_section_mentions(self, text: str) -> List[str]:
        found = []
        for m in _SECTION_REF.finditer(text):
            sec = self.normalize(m.group(1) or m.group(2))
            if sec and sec not in found:
                found.append(sec)
        return found
