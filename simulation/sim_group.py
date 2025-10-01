"""
SimGroup class implementation for the DDA model.

This module contains the SimGroup class which represents an experimental
group containing multiple phases.

"""

import numpy as np
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict
import pickle

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.stimulus import Stimulus
from simulation.SimPhase import SimPhase
from core.trial import Trial, ConfiguralCS


class SimGroup:
    """
    Represents an experimental group in the DDA model.
    
    A group contains multiple phases that are run sequentially.
    It manages the overall experimental design and coordinates
    learning across phases.
    
    Attributes:
        name (str): Name of the group
        phases (List[SimPhase]): List of phases in this group
        cues_map (Dict[str, Stimulus]): Map of all stimuli
        model: Reference to parent SimModel
    """
    
    def __init__(self, name: str, model: Any = None):
        """
        Initialize a SimGroup.
        
        Args:
            name: Name of the group
            model: Parent SimModel object
        """
        self.name = name
        self.model = model
        
        # Phases
        self.phases: List[SimPhase] = []
        self.no_of_phases = 0
        
        # Stimulus mappings
        self.cues_map: Dict[str, Stimulus] = {}
        self.all_stimuli: Set[str] = set()
        
        # Trial tracking
        self.total_trials = 0
        self.total_stimuli = 0
        self.phase_trials: List[int] = []
        
        # Configuration
        self.has_random = False
        self.no_of_combinations = 1
        
        # Results storage
        self.results: Dict[str, Any] = {}
        
        self.memory_maps: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Common element configuration
        self.common_map: Dict[str, Dict[str, float]] = {}
        self.shared_elements: Dict[str, List[Any]] = {}
        
        # Configural cue mappings
        self.config_cues_names: Dict[str, str] = {}
        self.config_cue_counter: int = 0

        # Control for threading
        self.control = None
        
        # Max duration tracking
        self.max_duration = 0
        self.max_micros = 0
        
        # Database key packing attributes
        self.hash_set = False
        self.bytes1 = self.bytes2 = self.bytes3 = self.bytes4 = self.bytes5 = 1
        self.s1 = self.s2 = self.s3 = self.s4 = self.s5 = 0
        self.trial_types = 0
        self.max_sessions = 1
        self.total_key_set = []

    def add_phase(self, phase_sequence: str, num_trials: Optional[int] = None) -> SimPhase:
        """
        Add a phase to this group.
        
        Args:
            phase_sequence: Trial sequence string
            num_trials: Number of trials (if not specified in sequence)
            
        Returns:
            Created SimPhase object
        """
        from simulation.SimPhase import SimPhase
        from simulation.configurables import ITIConfig, TimingConfiguration, ContextConfig
        from core.trial import Trial
        
        # Create default configs
        iti_config = ITIConfig()
        timing_config = TimingConfiguration()
        context_config = ContextConfig()
        
        # Create trial
        trial = Trial(phase_sequence, group=self)
        
        phase = SimPhase(
            phase_num=len(self.phases),
            sessions=1,
            seq=phase_sequence,
            order=[trial],
            stimuli2={phase_sequence: trial},
            sg=self,
            random_order=False,
            timing=timing_config,
            iti=iti_config,
            context=context_config,
            trials_in_all_phases=1,
            listed_stimuli=['A', 'B', '+'],
            varying_vartheta=False
        )
        
        if num_trials is not None:
            phase.trials = num_trials
            
        self.phases.append(phase)
        self.no_of_phases += 1
        
        # Update trial counts
        self.phase_trials.append(phase.trials)
        self.total_trials += phase.trials
        
        # Check for randomization
        if phase.random:
            self.has_random = True
            
        return phase
        
    def initialize(self):
        """Initialize the group for simulation."""
        # Collect all unique stimuli
        self._collect_all_stimuli()
        
        # Calculate total stimuli including context
        total_stimuli_with_context = self.total_stimuli
        if self.model and self.model.use_context:
            total_stimuli_with_context += 1  # Add context stimulus
        
        if self.model is not None:
            self.config_cues_names = getattr(self.model, "config_cues_names", {})
            self.config_cue_counter = getattr(self.model, "config_cue_counter", 0)
        if not self.config_cues_names:
            self.config_cues_names = {}
            self.config_cue_counter = 0
        
        # Create stimulus objects
        self._create_stimuli()
        
        if hasattr(self, '_materialize_configurals'):
            self._materialize_configurals()

        # Initialize phases
        for phase in self.phases:
            phase.initialize_stimuli()
            
        # Update all existing stimuli with correct total_stimuli count
        for stimulus in self.cues_map.values():
            stimulus.update_total_stimuli(total_stimuli_with_context)
            
        # Set up common elements if needed
        self._setup_common_elements()
        
        # Set total_max for all stimuli
        for stimulus in self.cues_map.values():
            if stimulus.total_max == 0:
                stimulus.total_max = 100  
        
        # CRITICAL: Set all stimulus names for all elements so they can predict any stimulus
        self._set_all_stimulus_names()
        
    def _set_all_stimulus_names(self):
        """Set all stimulus names for all elements so they can predict any stimulus."""
        # Get all stimulus names including context
        all_names = list(self.all_stimuli)
        if hasattr(self, 'context') and self.context:
            all_names.append('Context')  # Add context if it exists
        all_names.sort()
        
        
        # Set names for all stimuli so they can predict each other
        for stimulus in self.cues_map.values():
            stimulus.names = all_names.copy()
            
        # Set names for all elements in all stimuli
        for stimulus in self.cues_map.values():
            if hasattr(stimulus, 'get_list'):
                for element in stimulus.get_list():
                    if hasattr(element, 'set_all_stimulus_names'):
                        element.set_all_stimulus_names(all_names)
        
    def _collect_all_stimuli(self):
        """Collect all unique stimulus names from all phases."""
        # Skip if manually set
        if getattr(self, '_skip_collect_stimuli', False):
            return
            
        self.all_stimuli.clear()
        
        for phase in self.phases:
            for trial in phase.orderedSeq:
                for cs in trial.get_cues():
                    self.all_stimuli.add(cs.name)  # Extract name from CS object
                    if hasattr(self, "_ensure_configural_virtual") and isinstance(cs, ConfiguralCS) and cs.get_parts():
                        compound_name = cs.get_parts()
                        if cs.name not in self.config_cues_names:
                            self.config_cues_names[cs.name] = compound_name
                        self._ensure_configural_virtual(cs.name, compound_name)
                    
            # Add US if any reinforced trials
            reinforced_trials = [trial for trial in phase.orderedSeq if trial.is_reinforced()]
            print(f"DEBUG: Phase {phase.phaseNum} has {len(reinforced_trials)} reinforced trials out of {len(phase.orderedSeq)} total")
            for trial in reinforced_trials:
                print(f"DEBUG:   Reinforced trial: {trial.to_string()}")
            if any(trial.is_reinforced() for trial in phase.orderedSeq):
                self.all_stimuli.add('+')
                print(f"DEBUG: Added '+' stimulus to all_stimuli")
            else:
                print(f"DEBUG: No reinforced trials found, NOT adding '+' stimulus")
                
        self.total_stimuli = len(self.all_stimuli)
        print(f"DEBUG: Collected {self.total_stimuli} unique stimuli: {sorted(self.all_stimuli)}")
        
    def _create_stimuli(self):
        """Create Stimulus objects for all unique stimuli."""
        # Calculate total stimuli including context
        total_stimuli_with_context = self.total_stimuli
        if self.model and self.model.use_context:
            total_stimuli_with_context += 1  # Add context stimulus
        
        # Add context stimulus if enabled
        if self.model and self.model.use_context:
            # Create context stimulus (Ω)
            context = Stimulus(
                group=self,
                symbol='Context',
                alpha=self.model.context_alpha_r if self.model else 0.3,
                trials=self.total_trials,
                total_stimuli=total_stimuli_with_context  # Use correct total
            )
            print(f"DEBUG: Created Context stimulus with vartheta={context.vartheta}")
            
            # Set context-specific parameters
            context.set_alpha_r(self.model.context_alpha_r if self.model else 0.3)
            context.set_alpha_n(self.model.context_alpha_n if self.model else 0.3)
            context.set_salience(self.model.context_salience if self.model else 0.1)
            context.is_context = True
            context.set_max_duration(20)  # Match typical trial length
            context.set_all_max_duration(self.max_duration)
            
            # Add microstimuli
            context.add_microstimuli()
            
            self.cues_map['Context'] = context
            self.context = context  # Store reference
        
        for stim_name in self.all_stimuli:
            if stim_name not in self.cues_map:
                stimulus = Stimulus(
                    group=self,
                    symbol=stim_name,
                    alpha=self.model.alpha_r if self.model else 0.5,
                    trials=self.total_trials,
                    total_stimuli=total_stimuli_with_context  # Use correct total
                )
                print(f"DEBUG: Created stimulus '{stim_name}' with is_us={getattr(stimulus, 'is_us', 'NOT SET')}")
                
                # Set max duration based on phase configurations
                max_dur = self._get_max_duration_for_stimulus(stim_name)
                stimulus.set_max_duration(max_dur)
                stimulus.set_all_max_duration(self.max_duration)
                
                # Set alpha values from model
                stimulus.set_alpha_r(self.model.alpha_r if self.model else 0.5)
                stimulus.set_alpha_n(self.model.alpha_n if self.model else 0.5)
                
                # Set std parameter from phases (use first phase's std value)
                if self.phases and len(self.phases) > 0:
                    phase_std = getattr(self.phases[0], 'std', 1.0) 
                    stimulus.set_std(phase_std)
                
                # Preserve default CS parameters unless configured externally
                
                # Add microstimuli
                stimulus.add_microstimuli()
                
                self.cues_map[stim_name] = stimulus
        
        # Update all existing stimuli with correct total_stimuli count
        for stimulus in self.cues_map.values():
            if hasattr(stimulus, 'update_total_stimuli'):
                stimulus.update_total_stimuli(total_stimuli_with_context)
                
    def _get_max_duration_for_stimulus(self, stim_name: str) -> int:
        """
        Get maximum duration for a stimulus across all phases.
        
        Args:
            stim_name: Stimulus name
            
        Returns:
            Maximum duration
        """
        max_dur = 20
        
        # Check phase-specific configurations
        for phase in self.phases:
            if hasattr(phase, 'timing_config') and phase.timing_config:
                for trial_type, duration in phase.timing_config.items():
                    if stim_name in trial_type:
                        max_dur = max(max_dur, duration)
                        
        return max_dur
        
    def _setup_common_elements(self):
        """Set up common/shared elements between stimuli."""
        # Check for compound stimuli
        compounds = [s for s in self.all_stimuli if len(s) > 1 and s[0].isupper()]
        
        for compound in compounds:
            # Create common representation
            common_name = f"c{compound}"
            if common_name not in self.common_map:
                self.common_map[common_name] = {}
                
            # Set proportion of common elements
            # This is a simplified version - can be enhanced
            for elem in compound:
                if elem.isupper():
                    self.common_map[common_name][elem] = 0.2  # 20% common by default
                    
    def run(self) -> Dict[str, Any]:
        """
        Run all phases in this group.
        
        Returns:
            Dictionary of group results
        """
        group_results = {
            'group_name': self.name,
            'num_phases': self.no_of_phases,
            'total_trials': self.total_trials,
            'phase_results': []
        }
        
        # Initialize if not done
        if not self.cues_map:
            self.initialize()
            
        # Run each phase
        for phase_num, phase in enumerate(self.phases):
            # Set control if available
            if self.control:
                phase.control = self.control
                
            # Share stimulus objects with phase
            phase.cues = self.cues_map
            
            # Run phase
            phase_results = phase.run()
            group_results['phase_results'].append(phase_results)
            
            # Check for cancellation
            if self.control and hasattr(self.control, 'is_cancelled') and self.control.is_cancelled():
                break
                
        # Store final results
        self.results = group_results
        
        return group_results
        
    def get_results(self) -> Dict[str, Any]:
        """
        Get simulation results.
        
        Returns:
            Dictionary of results
        """
        if not self.results:
            return {}
            
        # Compile results
        compiled_results = {
            'group_name': self.name,
            'stimulus_weights': {},
            'trial_predictions': {},
            'phase_data': []
        }
        
        # Extract stimulus weights
        for name, stimulus in self.cues_map.items():
            if hasattr(stimulus, 'trial_w') and stimulus.trial_w is not None:
                compiled_results['stimulus_weights'][name] = stimulus.trial_w.tolist()
                
        # Extract phase-specific data
        for phase_num, phase in enumerate(self.phases):
            phase_data = {
                'phase_num': phase_num,
                'num_trials': phase.trials,
                'trial_sequence': phase.initial_seq,
                'random': phase.random
            }
            compiled_results['phase_data'].append(phase_data)
            
        return compiled_results
        
    # Memory/database methods (simplified versions)
    def make_map(self, key: str):
        """Create a memory map."""
        if key not in self.memory_maps:
            self.memory_maps[key] = {}
            
    def add_to_map(self, sub_key: str, value: Any, map_key: str, overwrite: bool = True):
        """Add to a memory map."""
        if map_key not in self.memory_maps:
            self.make_map(map_key)
            
        if overwrite or sub_key not in self.memory_maps[map_key]:
            self.memory_maps[map_key][sub_key] = value
            
    def get_from_db(self, sub_key: str, map_key: str) -> Any:
        """Get from memory map."""
        if map_key in self.memory_maps and sub_key in self.memory_maps[map_key]:
            return self.memory_maps[map_key][sub_key]
        return None
        
    def clear_map(self, map_key: str):
        """Clear a memory map."""
        if map_key in self.memory_maps:
            self.memory_maps[map_key].clear()
            
    def remove_map(self, map_key: str):
        """Remove a memory map."""
        if map_key in self.memory_maps:
            del self.memory_maps[map_key]
            
    def get_maps(self) -> Dict[str, Dict[str, Any]]:
        """Get all memory maps."""
        return self.memory_maps
        
    # Getters
    def get_name_of_group(self) -> str:
        """Get group name."""
        return self.name
        
    def get_no_of_phases(self) -> int:
        """Get number of phases."""
        return max(1, self.no_of_phases)  # Ensure at least 1
        
    def get_phases(self) -> List[SimPhase]:
        """Get list of phases."""
        return self.phases
        
    def get_cues_map(self) -> Dict[str, Stimulus]:
        """Get stimulus map."""
        return self.cues_map
        
    def get_model(self) -> Any:
        """Get parent model."""
        return self.model
        
    def get_total_trials(self) -> int:
        """Get total number of trials."""
        return self.total_trials
        
    def get_no_trial_types(self) -> int:
        """Get number of unique trial types."""
        trial_types = set()
        for phase in self.phases:
            for trial in phase.ordered_seq:
                trial_types.add(trial.get_trial_string())
        return len(trial_types)
        
    def get_trial_strings(self) -> List[str]:
        """Get all unique trial strings."""
        trial_types = set()
        for phase in self.phases:
            for trial in phase.ordered_seq:
                trial_types.add(trial.get_trial_string())
        return list(trial_types)
        
    def get_trial_type_index(self, trial_string: str) -> int:
        """Get index of a trial type."""
        trial_types = self.get_trial_strings()
        if trial_string in trial_types:
            return trial_types.index(trial_string)
        return -1
        
    def save_state(self, filename: str):
        """
        Save group state to file.
        
        Args:
            filename: Path to save file
        """
        state = {
            'name': self.name,
            'phases': len(self.phases),
            'results': self.results,
            'memory_maps': dict(self.memory_maps)
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(state, f)
            
    def load_state(self, filename: str):
        """
        Load group state from file.
        
        Args:
            filename: Path to load file
        """
        with open(filename, 'rb') as f:
            state = pickle.load(f)
            
        self.name = state['name']
        self.results = state['results']
        self.memory_maps = defaultdict(dict, state['memory_maps'])
    def get_cues_map(self) -> Dict[str, Stimulus]:
        """Get the cues map."""
        return self.cues_map
    
    def get_name_of_group(self) -> str:
        """Get the name of this group."""
        return self.name
    
    def get_model(self) -> Any:
        """Get the parent model."""
        return self.model
    
    def get_controller(self) -> Any:
        """Get the controller (returns self for simplicity)."""
        return self
    
    def get_no_of_phases(self) -> int:
        """Get number of phases."""
        return self.no_of_phases
    
    def get_first_occurrence(self, stimulus) -> int:
        """Get first occurrence phase of a stimulus (simplified)."""
        # For simplicity, return 0 for all stimuli
        return 0
    
    def compact_db(self):
        """Compact database (simplified)."""
        # For simplicity, do nothing
        pass

    def push_cache(self):
        """Push cache - placeholder method."""
        pass

    def initialize_db_keys(self):
        """Initialize database key packing parameters."""
        self.hash_set = False
        self.bytes1 = self.bytes2 = self.bytes3 = self.bytes4 = self.bytes5 = 1
        self.s1 = self.s2 = self.s3 = self.s4 = self.s5 = 0

        # Calculate maxima
        self.total_stimuli = max(1, len(self.cues_map))
        self.max_micros = max(1, self.max_micros)
        self.total_trials = max(1, self.total_trials)
        self.max_duration = max(1, self.max_duration)

        # Number of unique trial types
        trial_types = set()
        for phase in self.phases:
            for trial in phase.ordered_seq:
                trial_types.add(trial.get_trial_string())
        self.trial_types = max(1, len(trial_types))
        self.total_key_set = list(trial_types)

        # Calculate bit widths for packing
        self.bytes1 = self._get_c(self.total_stimuli * self.total_stimuli * self.no_of_phases)
        self.s1 = self.bytes1
        self.bytes2 = self._get_c(self.max_micros)
        self.s2 = self.bytes1 + self.bytes2 + 1
        self.bytes3 = self._get_c(self.total_trials)
        self.s3 = self.bytes1 + self.bytes2 + self.bytes3 + 2
        self.bytes4 = self._get_c(self.max_duration)
        self.s4 = self.bytes1 + self.bytes2 + self.bytes3 + self.bytes4 + 3
        self.bytes5 = self._get_c(1 * self.trial_types)  # maxSessions = 1
        self.s5 = self.bytes1 + self.bytes2 + self.bytes3 + self.bytes4 + self.bytes5 + 4
        self.hash_set = True

    def _get_c(self, val: int) -> int:
        """Calculate bits needed for a value."""
        if val in (0, 1):
            return 1
        else:
            import math
            return int(math.ceil(math.log(max(2, val), 2)))

    def create_db_key(self, se, trial_string: str, other, phase: int, session: int, trial: int, timepoint: int, isA: bool) -> int:
        """Create database key by packing fields."""
        if not self.hash_set:
            self.initialize_db_keys()

        # Get indices
        se_name_index = 0
        other_name_index = 0
        try:
            se_name_index = list(self.cues_map.keys()).index(se.get_name())
        except (ValueError, AttributeError):
            pass
        try:
            other_name_index = list(self.cues_map.keys()).index(other.get_name())
        except (ValueError, AttributeError):
            pass

        # Trial index
        trial_index = 0
        try:
            trial_index = self.total_key_set.index(trial_string)
        except (ValueError, IndexError):
            pass

        # Pack fields
        val = 1 if isA else 0
        key = val
        key = (key << self.bytes5) | ((session - 1) * self.trial_types + trial_index)
        key = (key << self.bytes4) | timepoint
        key = (key << self.bytes3) | trial
        key = (key << self.bytes2) | getattr(se, 'micro_index', 0)
        tail = (self.total_stimuli * self.total_stimuli * phase) + (self.total_stimuli * se_name_index) + other_name_index
        key = (key << self.bytes1) | tail
        return int(key)

    def get_names(self):
        """Get list of stimulus names."""
        return list(self.cues_map.keys())

    def phase_to_trial(self, phase: int) -> int:
        """Convert phase index to trial index."""
        trials = 0
        for i in range(phase + 1):
            if i < len(self.phases):
                phase_trials = getattr(self.phases[i], 'trials', 0)
                trials += max(1, phase_trials)
        return -1 if phase == -1 else trials - 1

    def trial_to_phase(self, trial: int) -> int:
        """Convert trial index to phase index."""
        phase_idx = 0
        for i in range(len(self.phases)):
            if self.phase_to_trial(i) >= trial:
                return phase_idx
            else:
                phase_idx += 1
        return phase_idx

    def get_first_occurrence(self, stimulus):
        """Get first occurrence phase of a stimulus."""
        if stimulus is None:
            return -1
        if hasattr(stimulus, 'get_name'):
            stim_name = stimulus.get_name()
            # Check if we have cached appearance
            if hasattr(self, 'appearance_list') and stim_name in self.appearance_list:
                return self.appearance_list[stim_name]

            # Search through phases
            for phase_idx, phase in enumerate(self.phases):
                # Check if stimulus appears in this phase
                if hasattr(phase, 'contains_stimulus') and phase.contains_stimulus(stimulus):
                    if not hasattr(self, 'appearance_list'):
                        self.appearance_list = {}
                    self.appearance_list[stim_name] = phase_idx
                    return phase_idx
                # Check context
                if hasattr(phase, 'get_context_config') and phase.get_context_config():
                    context_name = phase.get_context_config().get_symbol()
                    if context_name == stim_name:
                        if not hasattr(self, 'appearance_list'):
                            self.appearance_list = {}
                        self.appearance_list[stim_name] = phase_idx
                        return phase_idx

        return -1

    def add_trial_string(self, s: str):
        """Add trial string for tracking."""
        if not hasattr(self, 'trial_strings'):
            self.trial_strings = []
        if not hasattr(self, 'trial_strings2'):
            self.trial_strings2 = []

        parts = s.split('/')
        for s2 in parts:
            filtered = ''.join([c for c in s2 if not c.isdigit()])
            if filtered not in self.trial_strings:
                self.trial_strings.append(filtered)
            # Remove US symbol for trial_strings2
            filtered2 = filtered[:-1] if filtered and filtered[-1] in ['*', 'λ'] else filtered
            if filtered2 not in self.trial_strings2:
                self.trial_strings2.append(filtered2)

    def get_trial_strings(self):
        """Get trial strings."""
        if hasattr(self, 'trial_strings'):
            return self.trial_strings
        return []

    def get_trial_type_index(self, trial_string: str) -> int:
        """Get index of trial type."""
        trial_strings = self.get_trial_strings()
        if trial_string in trial_strings:
            return trial_strings.index(trial_string)
        return -1

    def db_contains(self, key: int, which_map: int = 1) -> bool:
        """Check if key exists in database."""
        if not hasattr(self, 'map'):
            return False
        return key in self.map if which_map == 1 else key in getattr(self, 'map2', {})

    def get_from_db_long(self, key: int, which_map: int = 1) -> float:
        """Get value from database by long key."""
        if not hasattr(self, 'map'):
            return 0.0
        return float(self.map.get(key, 0.0)) if which_map == 1 else float(getattr(self, 'map2', {}).get(key, 0.0))

    def get_db_keys(self, first: bool = True):
        """Get database keys."""
        if not hasattr(self, 'map'):
            return set()
        return set(self.map.keys()) if first else set(getattr(self, 'map2', {}).keys())

    def create_dbs(self):
        """Create databases (simplified)."""
        if not hasattr(self, 'map'):
            self.map = {}
        if not hasattr(self, 'map2'):
            self.map2 = {}

    def make_map(self, name: str):
        """Make a map."""
        if not hasattr(self, 'maps'):
            self.maps = {}
        if name not in self.maps:
            self.maps[name] = {}
        return self.maps[name]

    def compact_db(self):
        """Compact database (simplified)."""
        pass

    def close_dbs(self):
        """Close databases."""
        if hasattr(self, 'map'):
            self.map.clear()
        if hasattr(self, 'map2'):
            self.map2.clear()
        if hasattr(self, 'maps'):
            for map_name in self.maps:
                self.maps[map_name].clear()

    def db_contains(self, key: int, db: bool = True) -> bool:
        """Check if key exists in database."""
        if db:
            return key in getattr(self, 'map', {})
        else:
            return key in getattr(self, 'map2', {})

    def push_cache(self):
        """Push cache (placeholder)."""
        pass

        return self.no_of_phases
    
    def get_first_occurrence(self, stimulus) -> int:
        """Get first occurrence phase of a stimulus (simplified)."""
                                                  
        return 0
    
    def compact_db(self):
        """Compact database (simplified)."""
                                    
        pass

    def push_cache(self):
        """Push cache - placeholder method."""
        pass

                                                       
    def initialize_db_keys(self):
        """Initialize database key packing parameters."""
        self.hash_set = False
        self.bytes1 = self.bytes2 = self.bytes3 = self.bytes4 = self.bytes5 = 1
        self.s1 = self.s2 = self.s3 = self.s4 = self.s5 = 0

                          
        self.total_stimuli = max(1, len(self.cues_map))
        self.max_micros = max(1, self.max_micros)
        self.total_trials = max(1, self.total_trials)
        self.max_duration = max(1, self.max_duration)

                                      
        trial_types = set()
        for phase in self.phases:
            for trial in phase.ordered_seq:
                trial_types.add(trial.get_trial_string())
        self.trial_types = max(1, len(trial_types))
        self.total_key_set = list(trial_types)

                                          
        self.bytes1 = self._get_c(self.total_stimuli * self.total_stimuli * self.no_of_phases)
        self.s1 = self.bytes1
        self.bytes2 = self._get_c(self.max_micros)
        self.s2 = self.bytes1 + self.bytes2 + 1
        self.bytes3 = self._get_c(self.total_trials)
        self.s3 = self.bytes1 + self.bytes2 + self.bytes3 + 2
        self.bytes4 = self._get_c(self.max_duration)
        self.s4 = self.bytes1 + self.bytes2 + self.bytes3 + self.bytes4 + 3
        self.bytes5 = self._get_c(1 * self.trial_types)                   
        self.s5 = self.bytes1 + self.bytes2 + self.bytes3 + self.bytes4 + self.bytes5 + 4
        self.hash_set = True

    def _get_c(self, val: int) -> int:
        """Calculate bits needed for a value."""
        if val in (0, 1):
            return 1
        else:
            import math
            return int(math.ceil(math.log(max(2, val), 2)))

    def create_db_key(self, se, trial_string: str, other, phase: int, session: int, trial: int, timepoint: int, isA: bool) -> int:
        """Create database key by packing fields."""
        if not self.hash_set:
            self.initialize_db_keys()

                     
        se_name_index = 0
        other_name_index = 0
        try:
            se_name_index = list(self.cues_map.keys()).index(se.get_name())
        except (ValueError, AttributeError):
            pass
        try:
            other_name_index = list(self.cues_map.keys()).index(other.get_name())
        except (ValueError, AttributeError):
            pass

                     
        trial_index = 0
        try:
            trial_index = self.total_key_set.index(trial_string)
        except (ValueError, IndexError):
            pass

                     
        val = 1 if isA else 0
        key = val
        key = (key << self.bytes5) | ((session - 1) * self.trial_types + trial_index)
        key = (key << self.bytes4) | timepoint
        key = (key << self.bytes3) | trial
        key = (key << self.bytes2) | getattr(se, 'micro_index', 0)
        tail = (self.total_stimuli * self.total_stimuli * phase) + (self.total_stimuli * se_name_index) + other_name_index
        key = (key << self.bytes1) | tail
        return int(key)

    def get_names(self):
        """Get list of stimulus names."""
        return list(self.cues_map.keys())

    def phase_to_trial(self, phase: int) -> int:
        """Convert phase index to trial index."""
        trials = 0
        for i in range(phase + 1):
            if i < len(self.phases):
                phase_trials = getattr(self.phases[i], 'trials', 0)
                trials += max(1, phase_trials)
        return -1 if phase == -1 else trials - 1

    def trial_to_phase(self, trial: int) -> int:
        """Convert trial index to phase index."""
        phase_idx = 0
        for i in range(len(self.phases)):
            if self.phase_to_trial(i) >= trial:
                return phase_idx
            else:
                phase_idx += 1
        return phase_idx

    def get_first_occurrence(self, stimulus):
        """Get first occurrence phase of a stimulus."""
        if stimulus is None:
            return -1
        if hasattr(stimulus, 'get_name'):
            stim_name = stimulus.get_name()
                                                
            if hasattr(self, 'appearance_list') and stim_name in self.appearance_list:
                return self.appearance_list[stim_name]

                                   
            for phase_idx, phase in enumerate(self.phases):
                                                         
                if hasattr(phase, 'contains_stimulus') and phase.contains_stimulus(stimulus):
                    if not hasattr(self, 'appearance_list'):
                        self.appearance_list = {}
                    self.appearance_list[stim_name] = phase_idx
                    return phase_idx
                               
                if hasattr(phase, 'get_context_config') and phase.get_context_config():
                    context_name = phase.get_context_config().get_symbol()
                    if context_name == stim_name:
                        if not hasattr(self, 'appearance_list'):
                            self.appearance_list = {}
                        self.appearance_list[stim_name] = phase_idx
                        return phase_idx

        return -1

    def add_trial_string(self, s: str):
        """Add trial string for tracking."""
        if not hasattr(self, 'trial_strings'):
            self.trial_strings = []
        if not hasattr(self, 'trial_strings2'):
            self.trial_strings2 = []

        parts = s.split('/')
        for s2 in parts:
            filtered = ''.join([c for c in s2 if not c.isdigit()])
            if filtered not in self.trial_strings:
                self.trial_strings.append(filtered)
                                                 
            filtered2 = filtered[:-1] if filtered and filtered[-1] in ['*', 'λ'] else filtered
            if filtered2 not in self.trial_strings2:
                self.trial_strings2.append(filtered2)

    def get_trial_strings(self):
        """Get trial strings."""
        if hasattr(self, 'trial_strings'):
            return self.trial_strings
        return []

    def get_trial_type_index(self, trial_string: str) -> int:
        """Get index of trial type."""
        trial_strings = self.get_trial_strings()
        if trial_string in trial_strings:
            return trial_strings.index(trial_string)
        return -1

    def test_memory(self):
        """Test memory usage (simplified)."""
                                                                           
                                                                      
        pass

    def db_contains(self, key: int, which_map: int = 1) -> bool:
        """Check if key exists in database."""
        if not hasattr(self, 'map'):
            return False
        return key in self.map if which_map == 1 else key in getattr(self, 'map2', {})

    def get_from_db_long(self, key: int, which_map: int = 1) -> float:
        """Get value from database by long key."""
        if not hasattr(self, 'map'):
            return 0.0
        return float(self.map.get(key, 0.0)) if which_map == 1 else float(getattr(self, 'map2', {}).get(key, 0.0))

    def get_db_keys(self, first: bool = True):
        """Get database keys."""
        if not hasattr(self, 'map'):
            return set()
        return set(self.map.keys()) if first else set(getattr(self, 'map2', {}).keys())

    def set_rule(self, rule: int):
        """Set rule parameter."""
        self.rule = rule

    def get_rule(self) -> int:
        """Get rule parameter."""
        return getattr(self, 'rule', 2)

    def initialize_trial_arrays(self):
        """Initialize trial arrays for stimuli."""
        for stimulus in self.cues_map.values():
            if hasattr(stimulus, 'initialize_trial_arrays'):
                stimulus.initialize_trial_arrays()

    def get_no_trial_types(self) -> int:
        """Get number of trial types."""
        return len(self.get_trial_strings())

    def get_trial_type_index2(self, s: str) -> int:
        """Get index in trial_strings2."""
        if hasattr(self, 'trial_strings2') and s in self.trial_strings2:
            return self.trial_strings2.index(s)
        return -1

    def get_compound_indexes(self, compound: str):
        """Get compound indexes."""
        if not hasattr(self, 'trial_strings'):
            return []
        compound_indexes = []
        for i, s in enumerate(self.trial_strings):
            if compound in s and (len(s) == len(compound) or (abs(len(s) - len(compound)) == 1 and s.endswith('*'))):
                compound_indexes.append(i + 1)
        return compound_indexes

    def add_microstimuli(self):
                                                        
        shortest = 0
        for stimulus in self.cues_map.values():
            if hasattr(stimulus, 'get_all_max_duration'):
                shortest = max(shortest, stimulus.get_all_max_duration())
        
                                         
        for stimulus in self.cues_map.values():
            if hasattr(stimulus, 'add_microstimuli'):
                stimulus.add_microstimuli()
        
                                                
        self._setup_common_elements_between_stimuli()
    
    def _setup_common_elements_between_stimuli(self):
        """Set up common elements between stimuli (Java: common element logic)."""
                                                                       
                                                                           
                                                                                  
        
        for stimulus_name, stimulus in self.cues_map.items():
            for other_name, other_stimulus in self.cues_map.items():
                if stimulus_name != other_name:
                                                            
                    if hasattr(stimulus, 'set_common') and hasattr(other_stimulus, 'set_common'):
                                                         
                        common_value = 0.2                        
                        stimulus.set_common(other_name, common_value)
                        other_stimulus.set_common(stimulus_name, common_value)
    
    def get_total_max(self) -> float:
        """Get total maximum duration."""
        return float(self.max_duration)
    
    def get_sub_element_number(self) -> int:
        """Get sub-element number from model."""
        if self.model and hasattr(self.model, 'get_set_size'):
            return self.model.get_set_size()
        return 10                 
    
    def set_common(self, common: float):
        """Set common parameter."""
        self.common = common
    
    def get_common(self) -> float:
        """Get common parameter."""
        return getattr(self, 'common', 0.2)
    
    def set_control(self, control):
        """Set control for all phases."""
        self.control = control
        for phase in self.phases:
            if hasattr(phase, 'set_control'):
                phase.set_control(control)
    
    def set_model(self, model):
        """Set the model."""
        self.model = model
    
    def set_esther(self, esther: bool):
        """Set esther parameter."""
        self.esther = esther
    
    def set_no_of_phases(self, no_of_phases: int):
        """Set number of phases."""
        self.no_of_phases = no_of_phases
    
    def trial_count(self) -> int:
        """Get trial count."""
        count = 0
        for phase in self.phases:
            if hasattr(phase, 'is_random') and phase.is_random():
                multiplier = 1                                                     
            else:
                multiplier = 1
            if hasattr(phase, 'get_no_trials'):
                count += phase.get_no_trials() * multiplier
        return count
    
    def get_combination_no(self) -> int:
        """Get combination number."""
        if self.model and hasattr(self.model, 'get_combination_no'):
            return self.model.get_combination_no()
        return 1
    
    def get_sim_cues(self):
        """Get sim cues."""
        if not hasattr(self, 'sim_cues'):
            self.sim_cues = []
        return self.sim_cues
    
    def get_esther(self) -> bool:
        """Get esther parameter."""
        return getattr(self, 'esther', False)
    
    def set_shared_elements(self, data: dict):
        """Set shared elements."""
        if not hasattr(self, 'shared_map'):
            self.shared_map = {}
        self.shared_map.update(data)
        self.has_set = 1
    
    def set_total_trials(self, trials: int):
        """Set total trials."""
        self.total_trials = trials
    
    def get_total_trials(self) -> int:
        """Get total trials."""
        return self.total_trials
    
    def set_total_stimuli(self, stimuli: int):
        """Set total stimuli."""
        self.total_stimuli = stimuli
    
    def get_total_stimuli(self) -> int:
        """Get total stimuli."""
        return self.total_stimuli
    
    def initialize_db_keys(self):
        """Initialize database keys (Java: initializeDBKeys)."""
        self.max_sessions = 1
        self.total_key_set = []
        
        for i in range(self.no_of_phases):
            if i < len(self.phases):
                phase = self.phases[i]
                if hasattr(phase, 'get_ordered_seq'):
                    for trial in phase.get_ordered_seq():
                        trial_string = str(trial)
                        if trial_string not in self.total_key_set:
                            self.total_key_set.append(trial_string)
        
        self.trial_types = len(self.total_key_set)
        
                              
        self.bytes1 = self._get_c(self.total_stimuli * self.total_stimuli * self.no_of_phases)
        self.s1 = self.bytes1
        self.bytes2 = self._get_c(self.max_micros)
        self.s2 = self.bytes1 + self.bytes2 + 1
        self.bytes3 = self._get_c(self.total_trials)
        self.s3 = self.bytes1 + self.bytes2 + self.bytes3 + 2
        self.bytes4 = self._get_c(self.max_duration)
        self.s4 = self.bytes1 + self.bytes2 + self.bytes3 + self.bytes4 + 3
        self.bytes5 = self._get_c(self.max_sessions * self.trial_types)
        self.s5 = self.bytes1 + self.bytes2 + self.bytes3 + self.bytes4 + self.bytes5 + 4
    
    def get_c(self, val: int) -> int:
        """Calculate bits needed for a value (Java: getC)."""
        if val in (0, 1):
            return 1
        else:
            import math
            return int(math.ceil(math.log(max(2, val), 2)))
    
    def get_names(self):
        """Get names list."""
        if not hasattr(self, 'names'):
            self.names = list(self.cues_map.keys())
        return self.names
    
    def create_db_string(self, se, trial_string: str, other, phase: int, session: int, trial: int, timepoint: int, isA: bool) -> int:
        """Create database string (Java: createDBString)."""
        if not self.hash_set:
            self.initialize_db_keys()
            self.hash_set = True
        
                     
        se_name_index = 0
        other_name_index = 0
        try:
            se_name_index = self.get_names().index(se.get_name())
        except (ValueError, AttributeError):
            pass
        try:
            other_name_index = self.get_names().index(other.get_name())
        except (ValueError, AttributeError):
            pass
        
                     
        trial_index = 0
        try:
            trial_index = self.total_key_set.index(trial_string)
        except (ValueError, IndexError):
            pass
        
                     
        val = 1 if isA else 0
        key = val
        key = (key << self.bytes5) | ((session - 1) * self.trial_types + trial_index)
        key = (key << self.bytes4) | timepoint
        key = (key << self.bytes3) | trial
        key = (key << self.bytes2) | getattr(se, 'micro_index', 0)
        tail = (self.total_stimuli * self.total_stimuli * phase) + (self.total_stimuli * se_name_index) + other_name_index
        key = (key << self.bytes1) | tail
        return int(key)
    
    def get_db_keys(self, first: bool = True):
        """Get database keys."""
        if not hasattr(self, 'map'):
            return set()
        return set(self.map.keys()) if first else set(getattr(self, 'map2', {}).keys())
    
    def create_dbs(self):
        """Create databases (Java: createDBs)."""
                                      
        if not hasattr(self, 'map'):
            self.map = {}
        if not hasattr(self, 'map2'):
            self.map2 = {}
    
    def make_map(self, name: str):
        """Make a map (Java: makeMap)."""
        if not hasattr(self, 'maps'):
            self.maps = {}
        if name not in self.maps:
            self.maps[name] = {}
        return self.maps[name]
    
    def compact_db(self):
        """Compact database."""
                                        
        pass
    
    def close_dbs(self):
        """Close databases."""
                                     
        if hasattr(self, 'map'):
            self.map.clear()
        if hasattr(self, 'map2'):
            self.map2.clear()
        if hasattr(self, 'maps'):
            for map_name in self.maps:
                self.maps[map_name].clear()
    
    def db_contains(self, key: int, db: bool = True) -> bool:
        """Check if key exists in database."""
        if db:
            return key in getattr(self, 'map', {})
        else:
            return key in getattr(self, 'map2', {})
    
    def push_cache(self):
        """Push cache."""
                                  
        pass
    
    def phase_to_trial(self, phase: int, message: str = "") -> int:
        """Convert phase to trial (Java: phaseToTrial)."""
        trials = 0
        for i in range(phase + 1):
            if i < len(self.phases):
                phase_trials = getattr(self.phases[i], 'get_no_trials', lambda: 1)()
                trials += max(1, phase_trials)
        return -1 if phase == -1 else trials - 1
    
    def trial_to_phase(self, trial: int, message: str = "") -> int:
        """Convert trial to phase (Java: trialToPhase)."""
        phase = 0
        for i in range(len(self.phases)):
            if self.phase_to_trial(i, message) >= trial:
                return phase
            else:
                phase += 1
        return phase
    
    def add_to_db(self, key: int, entry: float):
        """Add to database (Java: addToDB)."""
        self.test_memory()
        if not hasattr(self, 'map'):
            self.map = {}
        self.map[key] = entry
    
    def add_to_map(self, key: str, entry: Any, map_name: str, add_now: bool = True):
        """Add to map (Java: addToMap)."""
        self.test_memory()
        if not hasattr(self, 'maps'):
            self.maps = {}
        if map_name not in self.maps:
            self.maps[map_name] = {}
        self.maps[map_name][key] = entry
    
    def add_to_db2(self, key: int, entry: float):
        """Add to database 2 (Java: addToDB2)."""
        self.test_memory()
        if not hasattr(self, 'map2'):
            self.map2 = {}
        self.map2[key] = entry
    
    def test_memory(self):
        """Test memory usage."""
                                   
        pass
    
    def get_key_set(self, map_name: str):
        """Get key set (Java: getKeySet)."""
        self.test_memory()
        if hasattr(self, 'maps') and map_name in self.maps:
            return set(self.maps[map_name].keys())
        return set()
    
    def get_from_db(self, key: str, map_name: str) -> Any:
        """Get from database (Java: getFromDB)."""
        self.test_memory()
        if hasattr(self, 'maps') and map_name in self.maps:
            return self.maps[map_name].get(key)
        return None
    
    def get_from_db_long(self, key: int) -> float:
        """Get from database by long key (Java: getFromDB)."""
        self.test_memory()
        if hasattr(self, 'map') and key in self.map:
            return self.map[key]
        return 0.0
    
    def reset_all(self):
        """Reset all (Java: resetAll)."""
        if not hasattr(self, 'trial_strings'):
            self.trial_strings = []
        if not hasattr(self, 'trial_strings2'):
            self.trial_strings2 = []
        if not hasattr(self, 'appearance_list'):
            self.appearance_list = {}
        if not hasattr(self, 'compound_appearance_list'):
            self.compound_appearance_list = {}
        
                           
        for stimulus in self.cues_map.values():
            if hasattr(stimulus, 'reset_completely'):
                stimulus.reset_completely()
        
                          
        for phase in self.phases:
            if hasattr(phase, 'reset'):
                phase.reset()
    
    def get_from_db2(self, key: int) -> float:
        """Get from database 2 (Java: getFromDB2)."""
        self.test_memory()
        if hasattr(self, 'map2') and key in self.map2:
            return self.map2[key]
        return 0.0
    
    def set_rule(self, rule: int):
        """Set rule (Java: setRule)."""
        self.rule = rule
    
    def initialize_trial_arrays(self):
        """Initialize trial arrays (Java: initializeTrialArrays)."""
        for stimulus in self.cues_map.values():
            if hasattr(stimulus, 'initialize_trial_arrays'):
                stimulus.initialize_trial_arrays()
    
    def get_no_trial_types(self) -> int:
        """Get number of trial types (Java: getNoTrialTypes)."""
        if hasattr(self, 'trial_strings'):
            return len(self.trial_strings)
        return 0
    
    def get_trial_strings(self):
        """Get trial strings (Java: getTrialStrings)."""
        if hasattr(self, 'trial_strings'):
            return self.trial_strings
        return []
    
    def get_first_occurrence(self, stimulus):
        """Get first occurrence (Java: getFirstOccurrence)."""
        if stimulus is None:
            return -1
        if hasattr(stimulus, 'get_name'):
            stim_name = stimulus.get_name()
            if hasattr(self, 'appearance_list') and stim_name in self.appearance_list:
                return self.appearance_list[stim_name]
        
                               
        for phase_idx, phase in enumerate(self.phases):
            if hasattr(phase, 'contains_stimulus') and phase.contains_stimulus(stimulus):
                if not hasattr(self, 'appearance_list'):
                    self.appearance_list = {}
                self.appearance_list[stim_name] = phase_idx
                return phase_idx
        
        return -1
    
    def add_trial_string(self, s: str):
        """Add trial string (Java: addTrialString)."""
        if not hasattr(self, 'trial_strings'):
            self.trial_strings = []
        if not hasattr(self, 'trial_strings2'):
            self.trial_strings2 = []
        
        parts = s.split('/')
        for s2 in parts:
            filtered = ''.join([c for c in s2 if not c.isdigit()])
            if filtered not in self.trial_strings:
                self.trial_strings.append(filtered)
                                                 
            filtered2 = filtered[:-1] if filtered and filtered[-1] in ['*', 'λ'] else filtered
            if filtered2 not in self.trial_strings2:
                self.trial_strings2.append(filtered2)
    
    def set_maximum_memory(self, memory: int):
        """Set maximum memory (Java: setMaximumMemory)."""
        self.memory = memory
    
    def get_trial_type_index(self, s: str) -> int:
        """Get trial type index (Java: getTrialTypeIndex)."""
        if hasattr(self, 'trial_strings') and s in self.trial_strings:
            return self.trial_strings.index(s)
        return -1
    
    def get_trial_type_index2(self, s: str) -> int:
        """Get trial type index 2 (Java: getTrialTypeIndex2)."""
        if hasattr(self, 'trial_strings2') and s in self.trial_strings2:
            return self.trial_strings2.index(s)
        return -1
    
    def get_compound_indexes(self, compound: str):
        """Get compound indexes (Java: getCompoundIndexes)."""
        if not hasattr(self, 'trial_strings'):
            return []
        compound_indexes = []
        for s in self.trial_strings:
            if compound in s and (len(s) == len(compound) or (abs(len(s) - len(compound)) == 1 and s.endswith('*'))):
                compound_indexes.append(self.get_trial_type_index(s) + 1)
        return compound_indexes
    
    def get_rule(self) -> int:
        """Get rule (Java: getRule)."""
        return getattr(self, 'rule', 2)

    def _ensure_configural_virtual(self, cue_name: str, compound: str):
        """Ensure a lowercase configural cue exists for the given compound."""
        if cue_name in self.cues_map:
            return
        virtual_name = cue_name
        if not virtual_name:
            if not self.config_cues_names:
                virtual_name = "a"
            else:
                while True:
                    next_char = chr(ord('a') + self.config_cue_counter)
                    self.config_cue_counter += 1
                    if next_char.isalpha() and next_char.islower() and next_char.upper() != next_char:
                        virtual_name = next_char
                        break
            self.config_cues_names[virtual_name] = compound
        stimulus = Stimulus(
            group=self,
            symbol=virtual_name,
            alpha=0.25,
            trials=self.total_trials,
            total_stimuli=max(1, self.total_stimuli)
        )
        self.cues_map[virtual_name] = stimulus
