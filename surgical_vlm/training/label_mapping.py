"""
Label Mapping Module
Converts between text labels (VLM output) and class indices (training loss).
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import yaml


class LabelMapper:
    """
    Maps text labels to class indices and vice versa for multi-task surgical video understanding.
    
    Handles fuzzy matching for VLM text outputs to canonical class names.
    """
    
    def __init__(self, config_dir: str = "configs/data"):
        self.config_dir = Path(config_dir)
        self._load_mappings()
    
    def _load_mappings(self):
        """Load canonical class names from config files."""
        
        # Phase mapping (Cholec80, HeiChole)
        self.phase_to_id = {
            "Preparation": 0,
            "CalotTriangleDissection": 1,
            "ClippingCutting": 2,
            "GallbladderDissection": 3,
            "GallbladderPackaging": 4,
            "CleaningCoagulation": 5,
            "GallbladderRetraction": 6,
            "Unknown": 7,
        }
        self.id_to_phase = {v: k for k, v in self.phase_to_id.items()}
        
        # Fuzzy matching aliases for phases
        self.phase_aliases = {
            "preparation": "Preparation",
            "prep": "Preparation",
            "calot triangle dissection": "CalotTriangleDissection",
            "calot dissection": "CalotTriangleDissection",
            "dissection": "CalotTriangleDissection",
            "clipping cutting": "ClippingCutting",
            "clipping and cutting": "ClippingCutting",
            "clipping": "ClippingCutting",
            "cutting": "ClippingCutting",
            "gallbladder dissection": "GallbladderDissection",
            "gallbladder packaging": "GallbladderPackaging",
            "packaging": "GallbladderPackaging",
            "cleaning coagulation": "CleaningCoagulation",
            "cleaning and coagulation": "CleaningCoagulation",
            "coagulation": "CleaningCoagulation",
            "cleaning": "CleaningCoagulation",
            "gallbladder retraction": "GallbladderRetraction",
            "retraction": "GallbladderRetraction",
        }
        
        # Instrument mapping (Cholec80, HeiChole)
        self.instrument_to_id = {
            "Grasper": 0,
            "Bipolar": 1,
            "Hook": 2,
            "Scissors": 3,
            "Clipper": 4,
            "Irrigator": 5,
            "SpecimenBag": 6,
        }
        self.id_to_instrument = {v: k for k, v in self.instrument_to_id.items()}
        
        self.instrument_aliases = {
            "grasper": "Grasper",
            "grasping forceps": "Grasper",
            "forceps": "Grasper",
            "bipolar": "Bipolar",
            "bipolar forceps": "Bipolar",
            "hook": "Hook",
            "electrocautery hook": "Hook",
            "cautery hook": "Hook",
            "scissors": "Scissors",
            "metzenbaum scissors": "Scissors",
            "clipper": "Clipper",
            "clip applier": "Clipper",
            "hemoclip": "Clipper",
            "irrigator": "Irrigator",
            "irrigation": "Irrigator",
            "suction": "Irrigator",
            "aspirator": "Irrigator",
            "specimen bag": "SpecimenBag",
            "endobag": "SpecimenBag",
            "retrieval bag": "SpecimenBag",
            "bag": "SpecimenBag",
        }
        
        # Action mapping (HeiChole)
        self.action_to_id = {
            "No_Action": 0,
            "Dissect": 1,
            "Grasp": 2,
            "Clip": 3,
            "Cut": 4,
            "Coagulate": 5,
            "Retract": 6,
            "Irrigate": 7,
            "Pack": 8,
            "Clean": 9,
        }
        self.id_to_action = {v: k for k, v in self.action_to_id.items()}
        
        self.action_aliases = {
            "no action": "No_Action",
            "idle": "No_Action",
            "dissect": "Dissect",
            "dissecting": "Dissect",
            "grasp": "Grasp",
            "grasping": "Grasp",
            "clip": "Clip",
            "clipping": "Clip",
            "cut": "Cut",
            "cutting": "Cut",
            "coagulate": "Coagulate",
            "coagulating": "Coagulate",
            "cauterize": "Coagulate",
            "retract": "Retract",
            "retracting": "Retract",
            "irrigate": "Irrigate",
            "irrigation": "Irrigate",
            "pack": "Pack",
            "packing": "Pack",
            "clean": "Clean",
            "cleaning": "Clean",
        }
        
        # CholecT50 triplet components
        self.triplet_instruments = [
            "Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator"
        ]
        self.triplet_verbs = [
            "No_Action", "Grasp", "Retract", "Dissect", "Coagulate",
            "Cut", "Clip", "Aspirate", "Irrigate", "Pack"
        ]
        self.triplet_targets = [
            "No_Target", "Gallbladder", "Cystic_Duct", "Cystic_Artery",
            "Hepatic_Duct", "Liver", "Peritoneum", "Fat", "Omentum",
            "Adhesion", "Clip", "Suture", "Needle", "Specimen_Bag", "Other"
        ]
    
    # ==================== PHASE MAPPING ====================
    
    def text_to_phase_id(self, text: str) -> int:
        """Convert phase text to class index with fuzzy matching."""
        if not text:
            return self.phase_to_id["Unknown"]
        
        text_clean = text.strip().lower()
        
        # Direct match (case insensitive)
        for canonical, idx in self.phase_to_id.items():
            if canonical.lower() == text_clean:
                return idx
        
        # Alias match
        for alias, canonical in self.phase_aliases.items():
            if alias in text_clean:
                return self.phase_to_id[canonical]
        
        # Partial match for compound names
        for canonical, idx in self.phase_to_id.items():
            if canonical.lower() in text_clean:
                return idx
        
        return self.phase_to_id["Unknown"]
    
    def phase_id_to_text(self, idx: int) -> str:
        """Convert phase index to canonical text."""
        return self.id_to_phase.get(idx, "Unknown")
    
    # ==================== INSTRUMENT MAPPING ====================
    
    def text_to_instrument_ids(self, text: str) -> List[int]:
        """Convert instrument text (comma-separated) to list of class indices."""
        if not text:
            return []
        
        ids = []
        # Split by comma or common separators
        parts = re.split(r'[,;|]', text)
        
        for part in parts:
            part = part.strip().lower()
            if not part:
                continue
            
            # Direct match
            matched = False
            for canonical, idx in self.instrument_to_id.items():
                if canonical.lower() == part:
                    ids.append(idx)
                    matched = True
                    break
            
            if matched:
                continue
            
            # Alias match
            for alias, canonical in self.instrument_aliases.items():
                if alias in part:
                    ids.append(self.instrument_to_id[canonical])
                    matched = True
                    break
            
            if matched:
                continue
            
            # Partial match
            for canonical, idx in self.instrument_to_id.items():
                if canonical.lower() in part:
                    ids.append(idx)
                    matched = True
                    break
        
        return list(set(ids))  # Deduplicate
    
    def instrument_ids_to_text(self, ids: List[int]) -> List[str]:
        """Convert instrument indices to canonical names."""
        return [self.id_to_instrument.get(i, "Unknown") for i in ids]
    
    # ==================== ACTION MAPPING ====================
    
    def text_to_action_id(self, text: str) -> int:
        """Convert action text to class index."""
        if not text:
            return self.action_to_id["No_Action"]
        
        text_clean = text.strip().lower()
        
        # Direct match
        for canonical, idx in self.action_to_id.items():
            if canonical.lower() == text_clean:
                return idx
        
        # Alias match
        for alias, canonical in self.action_aliases.items():
            if alias in text_clean:
                return self.action_to_id[canonical]
        
        # Partial match
        for canonical, idx in self.action_to_id.items():
            if canonical.lower() in text_clean:
                return idx
        
        return self.action_to_id["No_Action"]
    
    def action_id_to_text(self, idx: int) -> str:
        """Convert action index to canonical text."""
        return self.id_to_action.get(idx, "No_Action")
    
    # ==================== TRIPLET MAPPING (CholecT50) ====================
    
    def text_to_triplet_ids(self, text: str) -> Tuple[int, int, int]:
        """Convert triplet text 'Instrument, Verb, Target' to three indices."""
        if not text:
            return (0, 0, 0)
        
        parts = [p.strip() for p in text.split(',')]
        
        inst_id = 0
        verb_id = 0
        target_id = 0
        
        if len(parts) >= 1:
            inst_id = self._match_component(parts[0], self.triplet_instruments)
        if len(parts) >= 2:
            verb_id = self._match_component(parts[1], self.triplet_verbs)
        if len(parts) >= 3:
            target_id = self._match_component(parts[2], self.triplet_targets)
        
        return (inst_id, verb_id, target_id)
    
    def triplet_ids_to_text(self, inst_id: int, verb_id: int, target_id: int) -> str:
        """Convert three indices to triplet text."""
        inst = self.triplet_instruments[inst_id] if inst_id < len(self.triplet_instruments) else "Unknown"
        verb = self.triplet_verbs[verb_id] if verb_id < len(self.triplet_verbs) else "Unknown"
        target = self.triplet_targets[target_id] if target_id < len(self.triplet_targets) else "Unknown"
        return f"{inst}, {verb}, {target}"
    
    def _match_component(self, text: str, components: List[str]) -> int:
        """Fuzzy match a text to a component list."""
        text_clean = text.strip().lower()
        
        # Direct match
        for i, comp in enumerate(components):
            if comp.lower() == text_clean:
                return i
        
        # Partial match
        for i, comp in enumerate(components):
            if comp.lower() in text_clean or text_clean in comp.lower():
                return i
        
        return 0
    
    # ==================== BATCH PROCESSING ====================
    
    def batch_text_to_phase_ids(self, texts: List[str]) -> List[int]:
        return [self.text_to_phase_id(t) for t in texts]
    
    def batch_text_to_instrument_ids(self, texts: List[str]) -> List[List[int]]:
        return [self.text_to_instrument_ids(t) for t in texts]
    
    def batch_text_to_action_ids(self, texts: List[str]) -> List[int]:
        return [self.text_to_action_id(t) for t in texts]
    
    def batch_text_to_triplet_ids(self, texts: List[str]) -> List[Tuple[int, int, int]]:
        return [self.text_to_triplet_ids(t) for t in texts]


# Global instance
_label_mapper: Optional[LabelMapper] = None


def get_label_mapper(config_dir: str = "configs/data") -> LabelMapper:
    """Get or create global label mapper instance."""
    global _label_mapper
    if _label_mapper is None:
        _label_mapper = LabelMapper(config_dir)
    return _label_mapper


# Convenience functions
def text_to_phase_id(text: str) -> int:
    return get_label_mapper().text_to_phase_id(text)


def phase_id_to_text(idx: int) -> str:
    return get_label_mapper().phase_id_to_text(idx)


def text_to_instrument_ids(text: str) -> List[int]:
    return get_label_mapper().text_to_instrument_ids(text)


def instrument_ids_to_text(ids: List[int]) -> List[str]:
    return get_label_mapper().instrument_ids_to_text(ids)


def text_to_action_id(text: str) -> int:
    return get_label_mapper().text_to_action_id(text)


def action_id_to_text(idx: int) -> str:
    return get_label_mapper().action_id_to_text(idx)


def text_to_triplet_ids(text: str) -> Tuple[int, int, int]:
    return get_label_mapper().text_to_triplet_ids(text)


def triplet_ids_to_text(inst_id: int, verb_id: int, target_id: int) -> str:
    return get_label_mapper().triplet_ids_to_text(inst_id, verb_id, target_id)


if __name__ == "__main__":
    # Test the label mapper
    mapper = LabelMapper()
    
    # Test phase
    print("=== Phase Tests ===")
    test_phases = ["Clipping and Cutting", "preparation", "Calot triangle dissection", "Unknown phase"]
    for p in test_phases:
        idx = mapper.text_to_phase_id(p)
        back = mapper.phase_id_to_text(idx)
        print(f"  '{p}' -> {idx} -> '{back}'")
    
    # Test instruments
    print("\n=== Instrument Tests ===")
    test_inst = ["Grasper, Hook", "Bipolar forceps, scissors", "clip applier"]
    for inst in test_inst:
        ids = mapper.text_to_instrument_ids(inst)
        back = mapper.instrument_ids_to_text(ids)
        print(f"  '{inst}' -> {ids} -> {back}")
    
    # Test action
    print("\n=== Action Tests ===")
    test_actions = ["grasping", "cutting", "coagulating", "idle"]
    for a in test_actions:
        idx = mapper.text_to_action_id(a)
        back = mapper.action_id_to_text(idx)
        print(f"  '{a}' -> {idx} -> '{back}'")
    
    # Test triplet
    print("\n=== Triplet Tests ===")
    test_triplets = [
        "Grasper, Grasp, Gallbladder",
        "Hook, Dissect, Cystic Duct",
        "Clipper, Clip, Cystic Artery",
    ]
    for t in test_triplets:
        ids = mapper.text_to_triplet_ids(t)
        back = mapper.triplet_ids_to_text(*ids)
        print(f"  '{t}' -> {ids} -> '{back}'")