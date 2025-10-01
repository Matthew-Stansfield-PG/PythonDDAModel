from typing import Optional


class CS:
    TOTAL = None
    CS_TOTAL = None
    OMEGA = None
    US = None
    
    def __init__(self, name: str, hash_code: int = 0, group: int = 0, 
                 is_probe: bool = False, string_pos: int = 0):
        self.name = name
        self.group = group
        self.string_pos = string_pos
        self.is_probe = is_probe
        self.show_primes = False
        self.trial_string = ""
        self.hash_code = hash_code if hash_code else id(self)
        
    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return f"CS(name='{self.name}', hash={self.hash_code}, group={self.group})"
    
    def __eq__(self, other) -> bool:
        if isinstance(other, CS):
            return (self.name == other.name and 
                    self.hash_code == other.hash_code and
                    self.group == other.group and
                    self.string_pos == other.string_pos)
        return self.name == str(other)
    
    def __lt__(self, other) -> bool:
        if not isinstance(other, CS):
            return NotImplemented
            
        if self.name != other.name:
            return self.name < other.name
        if self.group != other.group:
            return self.group < other.group
        if self.hash_code != other.hash_code:
            return self.hash_code < other.hash_code
        return self.string_pos < other.string_pos
    
    def __hash__(self) -> int:
        return self.hash_code
    
    def copy(self) -> 'CS':
        new_cs = CS(self.name, self.hash_code, self.group, 
                   self.is_probe, self.string_pos)
        new_cs.show_primes = self.show_primes
        new_cs.trial_string = self.trial_string
        return new_cs
    
    def is_compound(self) -> bool:
        return len(self.name) > 1
    
    def is_configural(self) -> bool:
        return len(self.name) == 1 and self.name.islower()
    
    def is_serial_configural(self) -> bool:
        return False
    
    def get_probe_symbol(self) -> str:
        num_primes = self.hash_code + 1 if self.show_primes else 0
        primes = "'" * num_primes
        return f"{self.name}{primes}"
    
    def get_local_string_pos(self) -> int:
        if not self.trial_string:
            return 0
            
        pos = 0
        count = -1
        
        for i, char in enumerate(self.trial_string):
            if char == self.name:
                count += 1
                if count == self.hash_code:
                    return i + 1
                    
        return pos
    
    def get_name(self) -> str:
        return self.name
    
    def set_name(self, name: str):
        self.name = name
        
    def get_group(self) -> int:
        return self.group
        
    def set_group(self, group: int):
        self.group = group
        
    def get_hash_code(self) -> int:
        return self.hash_code
        
    def set_hash_code(self, code: int):
        self.hash_code = code
        
    def get_string_pos(self) -> int:
        return self.string_pos
        
    def set_string_pos(self, pos: int):
        self.string_pos = pos
        
    def get_trial_string(self) -> str:
        return self.trial_string
        
    def set_trial_string(self, trial_string: str):
        self.trial_string = trial_string
        
    def set_probe(self, is_probe: bool):
        self.is_probe = is_probe
        
    def set_show_primes(self, show: bool):
        self.show_primes = show


CS.TOTAL = CS("Total", 0, 0)
CS.CS_TOTAL = CS("CS_Total", 0, 0)
CS.OMEGA = CS("Omega", 0, 0)
CS.US = CS("US", 0, 0)


class ConfiguralCS(CS):
    SERIAL_SEP = "->"
    
    def __init__(self, name: str, hash_code: int = 0, group: int = 0,
                 parts: str = "", is_serial: bool = False):
        super().__init__(name, hash_code, group)
        self.parts = parts
        self.is_serial = is_serial
        
    def get_parts(self) -> str:
        return self.parts
    
    def set_parts(self, parts: str):
        self.parts = parts
        
    def is_serial_configural(self) -> bool:
        return self.is_serial
    
    def __repr__(self) -> str:
        serial_str = ", serial=True" if self.is_serial else ""
        return f"ConfiguralCS(name='{self.name}', parts='{self.parts}'{serial_str})"