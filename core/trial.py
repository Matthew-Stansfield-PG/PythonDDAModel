from typing import Set, Dict, Optional, Any
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.cs import CS, ConfiguralCS


class Trial:
    def __init__(self, trial_string: str, is_probe: bool = False, 
                 string_index: int = 0, sel_stim: str = "", 
                 string_pos: int = 0, trial_number: int = 0,
                 group: Optional[Any] = None):
        self.trial_string = trial_string
        self.is_probe = is_probe
        self.trial_number = trial_number
        self.cues: Set[CS] = set()
        self.group = group
        
        self._parse_trial_string(string_index, sel_stim, string_pos)
        
    def _parse_trial_string(self, string_index: int, sel_stim: str = "", string_pos: int = 0):
        cues_with_configurals = sel_stim if sel_stim else self.trial_string
        
        for char in self.trial_string:
            if char.islower() or self._is_context(char):
                cues_with_configurals += char
                
        cs_chars = list(cues_with_configurals)
        probe_count: Dict[str, int] = {}
        compound = ""
        index = 0
        
        for i, char in enumerate(cs_chars):
            if self._is_context(char):
                context = CS(char, 0, 0)
                self.cues.add(context)
                
            elif char.isalpha() and char.isupper():
                count = probe_count.get(char, 0)
                compound += char
                
                if not self._timing_per_trial():
                    string_pos = 0
                    count = 0
                
                probe_cs = False
                try:
                    probe_cs = cs_chars[i + 1] == '^'
                except IndexError:
                    pass
                
                cs = CS(char, 0, 0)
                self.cues.add(cs)
                count += 1
                probe_count[char] = count
                
            elif char.isalpha() and not char.isupper():
                cs = CS(char, 0, 0)
                self.cues.add(cs)
                
            elif char == '+':
                cs = CS('+', 0, 0)
                self.cues.add(cs)
                
            if char.isalpha() and not self._is_context(char) and self._timing_per_trial():
                index += 1
            string_pos += 1

                                                                             
        added = self._add_configural_cues(cues_with_configurals)
        if added:
            config_name, compound_parts = added
            self._register_configural_mapping(config_name, compound_parts)
            
    def _timing_per_trial(self) -> bool:
        return False
    
    def _is_context(self, char: str) -> bool:
        return char in ['Ω', 'ω', 'Omega', 'omega', 'C', 'c', 'CTX', 'ctx']
            
    def _should_add_configurals(self, cue_string: str) -> bool:
        elemental_cues = [c for c in cue_string if c.isupper()]
        return len(elemental_cues) > 1
    
    def _add_configural_cues(self, cue_string: str):
        elemental_cues = [c for c in cue_string if c.isupper()]
        
        if len(elemental_cues) > 1:
            config_name = elemental_cues[0].lower()
            compound_parts = ''.join(elemental_cues)
            
            if not any(cs.get_name() == config_name for cs in self.cues):
                config_cs = ConfiguralCS(config_name, 0, 0, parts=compound_parts)
                self.cues.add(config_cs)
                return config_name, compound_parts
        return None
    
    def _register_configural_mapping(self, config_name: str, compound: str) -> None:
        group = getattr(self, 'group', None)
        if group is None and hasattr(self, '_group_ref'):
            group = self._group_ref()
        if group is None:
            return
        if not hasattr(group, 'config_cues_names'):
            return
        if config_name not in group.config_cues_names:
            group.config_cues_names[config_name] = compound
    
    def _is_context(self, char: str) -> bool:
        return char in ['Ω', 'ω', 'Omega', 'omega', 'C', 'c']
    
    def copy(self) -> 'Trial':
        new_trial = Trial(self.trial_string, self.is_probe, 0, group=self.group)
        new_trial.set_cues(self.cues)
        new_trial.set_trial_number(self.trial_number)
        return new_trial
    
    def is_reinforced(self) -> bool:
        return self.trial_string.endswith('+')
    
    def get_probe_symbol(self) -> str:
        probe_name = "("
        
        for char in self.trial_string:
            if char.isupper():
                probe_name += char
                
        probe_name += "+" if self.is_reinforced() else "-"
        probe_name += ")"
        
        primes = "'" * self.trial_number
        return f"{probe_name}{primes}"
    
    def get_cues(self) -> Set[CS]:
        return self.cues
    
    def set_cues(self, cues: Set[CS]):
        self.cues = cues.copy()
        
    def get_trial_string(self) -> str:
        return self.trial_string
    
    def set_trial_string(self, trial_string: str):
        self.trial_string = trial_string
        self.cues.clear()
        self._parse_trial_string(0)
        
    def set_probe(self, is_probe: bool):
        self.is_probe = is_probe
        
    def get_trial_number(self) -> int:
        return self.trial_number
    
    def set_trial_number(self, number: int):
        self.trial_number = number
        
    def is_reinforced(self) -> bool:
        return "+" in self.trial_string
    
    def to_string(self) -> str:
        return self.trial_string
    
    def __str__(self) -> str:
        return self.trial_string
    
    def __repr__(self) -> str:
        probe_str = ", probe=True" if self.is_probe else ""
        return f"Trial('{self.trial_string}', number={self.trial_number}{probe_str})"