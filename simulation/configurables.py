from typing import Dict, List, Set, Any, Optional, Tuple
import random
import sys
from pathlib import Path

                                          
sys.path.insert(0, str(Path(__file__).parent.parent))


class TimingConfiguration:
    """
    Timing configuration class
    Handles stimulus timing and onset/offset calculations.
    """
    
    def __init__(self):
        """Initialize timing configuration."""
        self.timings: Dict[str, Tuple[int, int]] = {}
        self.default_cs_onset = 0
        self.default_cs_offset = 5
                                                              
        self.default_us_onset = 1                                    
        self.default_us_offset = 5                                            
        
    def make_timings(self, trials: List) -> 'TimingResult':
        """
        Create timing configuration for trials.
        
        Args:
            trials: List of trial objects or CS objects
            
        Returns:
            TimingResult object with timing information
        """
                                                          
        timings = {}
        
                                                     
        has_us = False

                                         
        for cs in trials:
            if hasattr(cs, 'get_name'):
                cs_name = cs.get_name()
                if cs_name == '+':      
                    has_us = True
                    timings[cs] = (self.default_us_onset, self.default_us_offset)
                else:      
                    timings[cs] = (self.default_cs_onset, self.default_cs_offset)
        
        from core.cs import CS
        timings[CS.TOTAL] = (self.default_cs_onset, self.default_cs_offset)
        if has_us:
            timings[CS.US] = (self.default_us_onset, self.default_us_offset)
        else:
            timings[CS.US] = (-1, -1)
        timings[CS.CS_TOTAL] = (self.default_cs_onset, self.default_cs_offset)
        
        return TimingResult(timings)
    
    def restart_onsets(self):
        """Restart onset calculations."""
        pass
    
    def set_trials(self, trials: List):
        """Set trials for timing configuration."""
        pass
    
    def get_total(self) -> Tuple[int, int]:
        """Get total timing range."""
        return (0, 20)                                           
    
    def get_us(self) -> Tuple[int, int]:
        """Get US timing."""
        return (self.default_us_onset, self.default_us_offset)
    
    def get_cs_total(self) -> Tuple[int, int]:
        """Get CS total timing."""
        return (self.default_cs_onset, self.default_cs_offset)
    
    def clear_defaults(self):
        """Clear default timing values."""
        pass


class TimingResult:
    """Result object for timing calculations."""
    
    def __init__(self, timings: Dict[str, Tuple[int, int]]):
        """Initialize with timing data."""
        self.timings = timings
    
    def get(self, key: Any, default: Any = None) -> Any:
        """Get timing for a key."""
        return self.timings.get(key, default)
    
    def __contains__(self, key: Any) -> bool:
        """Check if key exists in timings."""
        return key in self.timings
    
    def get_total(self) -> Tuple[int, int]:
        """Get total timing range."""
        from core.cs import CS
        return self.timings.get(CS.TOTAL, (0, 20))
    
    def get_us(self) -> Tuple[int, int]:
        """Get US timing."""
        from core.cs import CS
        return self.timings.get(CS.US, (self.timings.get(CS.TOTAL, (0, 20))[0],
                                        self.timings.get(CS.TOTAL, (0, 20))[1]))
    
    def get_cs_total(self) -> Tuple[int, int]:
        """Get CS total timing."""
        from core.cs import CS
        return self.timings.get(CS.CS_TOTAL, (0, 20))


class ITIConfig:
    """
    ITI (Inter-Trial Interval) configuration class.
    Handles ITI duration and timing.
    """
    
    def __init__(self):
        """Initialize ITI configuration."""
        self.iti_duration = 3
        self.minimum = 3
        self.maximum = 3
        self.trials = []
        
    def set_trials(self, trials: List):
        """Set trials for ITI calculation."""
        self.trials = trials
        
    def get_minimum(self) -> int:
        """Get minimum ITI duration."""
        return self.minimum
        
    def get_maximum(self) -> int:
        """Get maximum ITI duration."""
        return self.maximum
        
    def next(self) -> int:
        """Get next ITI duration."""
        return self.iti_duration
        
    def reset(self):
        """Reset ITI configuration."""
        pass


class ContextConfig:
    """
    Context configuration class.
    Handles context stimulus configuration.
    """
    
    def __init__(self, symbol: str = "Context"):
        """Initialize context configuration."""
        self.symbol = symbol
        
    def get_symbol(self) -> str:
        """Get context symbol."""
        return self.symbol
        
    def to_string(self) -> str:
        """Get string representation."""
        return self.symbol
        
    def get_context(self) -> 'Context':
        """Get context object."""
        return Context(self.symbol)


class Context:
    """Context object for context configuration."""
    
    def __init__(self, symbol: str):
        """Initialize context."""
        self.symbol = symbol
        
    def to_string(self) -> str:
        """Get string representation."""
        return self.symbol
