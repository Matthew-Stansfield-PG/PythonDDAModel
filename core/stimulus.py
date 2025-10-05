

import numpy as np
from typing import List, Dict, Optional, Set, Any, Tuple
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.SimElement import StimulusElement


class Stimulus:
    """
    Represents a complete stimulus in the DDA model.
    
    A Stimulus is composed of multiple StimulusElements that represent
    the temporal extent of the stimulus. This class manages the collection
    of elements and coordinates their learning and activation.

    """
    
    def __init__(self, group: Any, symbol: str, alpha: float = 0.5,
                 trials: int = 100, total_stimuli: int = 10):
        self.group = group
        self.symbol = symbol
        self.trials = trials
        self.total_stimuli = total_stimuli
        
                               
        self.alpha_r = alpha
        self.alpha_n = alpha
        self.beta = 0.1                
        self.salience = 1.0                
        self.omicron = 0.1                
        
                                                    
        self.tau1 = 0.99995                                              
        self.tau2 = 0.9999                                       
        self.vartheta = 0.5                         
        self.std = 1.0                                                           
        
                                          
        self.cues: List[StimulusElement] = []
        self.max_duration = 0
        self.total_max = 0
        
                                          
        self.presence_trace = 0.0
        self.active = False
        self.has_been_active = False
        self.timepoint = 0
        self.onset = 0
        self.last_onset = 0
        self.last_offset = 0
        self.current_duration = 0.0
        self.duration_point = 0
        
                        
        self.trial_count = 0
        self.current_phase = 0
        
                                                           
        self.not_divided = True
        self.trial_timepoints_counter = 0
        
                                       
        self.names: List[str] = []
        self.average_weights: Optional[np.ndarray] = None
        self.average_weights_a: Optional[np.ndarray] = None
        self.average_average_weights: Optional[np.ndarray] = None
        self.average_average_weights_a: Optional[np.ndarray] = None
        
                                
        self.trial_w: Optional[np.ndarray] = None
        self.trial_wa: Optional[np.ndarray] = None
        self.delta_ws: Optional[np.ndarray] = None
        self.asymptotes: Optional[np.ndarray] = None
        
                           
        self.activity: Optional[np.ndarray] = None
        self.was_active_last: Optional[np.ndarray] = None
        
                                    
        self.common_map: Dict[str, 'Stimulus'] = {}
        
                     
        self.is_us = "US" in symbol or "+" in symbol
        self.is_context = "context" in symbol.lower() or symbol in ['CTX', 'ctx']
        self.is_cs = not self.is_us and not self.is_context
        self.is_probe = False
        self.disabled = False
        self.micros_set = False
        
                             
        self.lambda_plus = 1.0
        self.us_boost = 0.08
        self.csc_like = 2.0                          
        self.presence_mean = True                                                                   
        self.reset_context = True
        self.context_reset = 0.95
        
                           
        self._initialize_arrays()
        
    def _initialize_arrays(self):
        """Initialize storage arrays."""
        if self.trials > 0 and self.total_stimuli > 0:
            self.average_average_weights = np.zeros(self.total_stimuli)
            self.average_average_weights_a = np.zeros(self.total_stimuli)
            self.average_prediction = np.zeros(self.trials)
            self.average_error = np.zeros(self.trials)
            
            if hasattr(self.group, 'get_no_of_phases'):
                num_phases = max(1, self.group.get_no_of_phases())
                self.was_active_last = np.zeros(num_phases, dtype=bool)
                self.activity = np.zeros((num_phases, self.trials + 1), dtype=bool)
                
    def initialize_trial_arrays(self):
        """Initialize trial-related arrays."""
        if hasattr(self.group, 'get_no_of_phases') and hasattr(self.group, 'get_no_trial_types'):
            num_phases = max(1, self.group.get_no_of_phases())
            num_trial_types = max(1, self.group.get_no_trial_types()) + 1
            
            if self.trial_w is None:
                self.trial_w = np.zeros((num_trial_types, num_phases, self.trials, self.total_stimuli))
                self.trial_wa = np.zeros((num_trial_types, num_phases, self.trials, self.total_stimuli))
                
            if hasattr(self.group, 'max_iti'):
                max_iti = self.group.max_iti
            else:
                max_iti = 50               
                
            if self.delta_ws is None and self.total_max > 0:
                self.delta_ws = np.zeros((num_phases, self.trials, self.total_max + max_iti))
                self.asymptotes = np.zeros((num_phases, self.trials, self.total_max + max_iti))
                
    def add_microstimuli(self):
        """
        Create and add microstimulus elements.
        
        """
        if self.max_duration <= 0:
            self.max_duration = 1
            
                                                                         
        num_elements = int(self.max_duration) if not self.disabled else 1
        
        if not self.micros_set:
            self.cues = []
            
            for i in range(num_elements):
                element = StimulusElement(
                    micro_index=i,
                    parent=self,
                    group=self.group,
                    name=self.symbol,
                    alpha=self.alpha_r / self.max_duration if self.max_duration > 0 else self.alpha_r,
                    std=self.std,
                    trials=self.trials,
                    total_stimuli=self.total_stimuli,
                    total_micros=num_elements,
                    total_max=self.total_max if self.total_max > 0 else 100,
                    generalization=1.0,
                    lambda_plus=self.lambda_plus,
                    us_boost=self.us_boost,
                    vartheta=self.vartheta,
                    presence_mean=self.presence_mean,
                )
                
                                                            
                                                                          
                
                                                                  
                element.setSubsetSize(10)                          
                
                                            
                element.initialize_weights(
                    subelement_number=10,                       
                    total_stimuli=self.total_stimuli,
                    total_max=self.total_max if self.total_max > 0 else 100
                )
                
                                        
                element.setNAlpha(self.alpha_n)
                element.setRAlpha(self.alpha_r)
                element.setSalience(self.salience)                                
                element.setBeta(self.beta)
                element.setDisabled(self.disabled)
                element.setCSCLike(self.csc_like)
                
                self.cues.append(element)
                
            self.micros_set = True
            
                                                         
            self.initialize_trial_arrays()
            
    def update_presence_trace(self, duration: float):
        self.current_duration = duration
        
                                                                                          
        if self.presence_trace < 0.01 and self.timepoint == 0:
            self.start_decay = False
            
                                                              
        if self.is_active():
            self.has_been_active = True
            
                                                            
        if self.is_active() and self.onset == 0 and not getattr(self, 'start_decay', False):
            self.presence_trace = 1.0
            self.start_decay = True
        elif self.is_active() and getattr(self, 'start_decay', False):
            self.presence_trace *= self.tau1
        elif not self.is_active() and self.has_been_active and getattr(self, 'start_decay', False):
            self.presence_trace *= self.tau2
            self.onset = 0
            
                                                                       
        if not self.is_active() and not self.has_been_active:
            self.presence_trace = 0.0
            
                                                                    
        if self.presence_trace > 1:
            self.presence_trace = 1.0
        
    def set_trial_length(self, trial_length: int):
        self.trial_length = trial_length
                                             
        for elem in self.cues:
            elem.setTrialLength(trial_length)
    
    def set_std(self, std: float):
        self.std = std
                                    
        for elem in self.cues:
            elem.std = std
                                                  
            micro_plus = elem.micro_index + 1
            adj = 0                      
            ctxratio = getattr(elem, 'ctxratio', 1.0)
            
            if elem.is_us:
                                                        
                elem.denominator = 2.0 * (micro_plus + adj) * elem.uscv * ctxratio
            else:
                                                       
                elem.denominator = 2.0 * (micro_plus + adj) * elem.std * ctxratio
    
    def set_duration(self, dur: int, onset: int, offset: int, duration_point: int, active: bool, real_time: int):                                                              
        self.duration_point = duration_point - 1
        self.active = active

                                                            
        if self.symbol in ['A', 'B', '+'] and active:
            print(f"DEBUG {self.symbol}: set_duration called with active={active}, duration_point={duration_point}, onset={onset}, offset={offset}, real_time={real_time}")

                                                     
        if offset > 0:
            self.last_offset = offset
        if onset != -1:
            self.last_onset = onset

                                                                     
        if not self.is_context and real_time > self.last_onset + 1 and real_time <= self.last_offset + 2:
            self.update = True
        elif not self.is_context:
            self.update = False
        elif self.is_context and real_time > self.last_onset + 1 and real_time <= self.last_offset + 1:
            self.update = True
        elif self.is_context:
            self.update = False
            
                                                           
        self.update_presence_trace(offset - onset)
        
                                                      
        if self.symbol in ['A', 'B', '+'] and active:
            print(f"DEBUG {self.symbol}: after update_presence_trace, has_been_active={self.has_been_active}")
        
                                                                                                
        update_duration = self.max_duration if self.current_duration == 0 else self.current_duration
        
                                                 
        for elem in self.cues:
            elem.setActive(self.symbol, active, duration_point)
            
                                    
                                                                                          
        update_duration = self.current_duration if self.current_duration > 0 else self.max_duration

        for n, elem in enumerate(self.cues):
            presence = (self.presence_trace *
                        getattr(self, 'presence_max', 1.0)) if n < update_duration else 0
            elem.updateActivation(self.symbol, presence, self.current_duration, n)
            
            
    def increment_timepoint(self, time: int, iti: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        if self.symbol in ['A', 'B', '+'] and time <= 10:
            print(f"DEBUG {self.symbol}: increment_timepoint called at time={time}, iti={iti}")
                                     
        if not hasattr(self, 'names') or not self.names:
            self.names = list(self.group.get_cues_map().keys())
        
                                         
        if not hasattr(self, 'total_max') or self.total_max == 0:
            self.total_max = 5                                  
        
                                      
        if self.average_weights is None:
            self.average_weights = np.zeros(self.total_stimuli)
        if self.average_weights_a is None:
            self.average_weights_a = np.zeros(self.total_stimuli)
        
                                                                                                                         
        old_v = self.average_weights[self.names.index("+")] if self.average_weights is not None and "+" in self.names else 0
        
                                                                              
                                                                                 
        
                                                                  
        if self.average_average_weights_a is None:
            self.average_average_weights_a = np.zeros(self.total_stimuli)
        temp_map = np.zeros((self.total_stimuli, self.total_max))
        delta_w = 0.0
        asymptote_mean = 0.0
        
                                                                     
        for element in self.cues:
            asymptote_mean += element.get_dynamic_asymptote() / len(self.cues)
            key = element.incrementTimepoint(time)
            obj = self.group.get_from_db(str(self.current_phase), key)
            
            if obj is not None:
                                                                  
                if len(obj.shape) == 3:
                                                                                                              
                    temp_map = np.sum(obj, axis=0)
                    if temp_map.shape != (self.total_stimuli, self.total_max):
                                                       
                        temp_map = temp_map[:self.total_stimuli, :self.total_max]
                else:
                    temp_map = obj
            else:
                                                                                
                if hasattr(element, 'subelement_weights') and element.subelement_weights is not None:
                                                                
                    temp_map = np.zeros((self.total_stimuli, self.total_max))
                    for i in range(self.total_stimuli):
                        for j in range(self.total_max):
                            if (i < element.subelement_weights.shape[1] and 
                                j < element.subelement_weights.shape[2]):
                                                                                              
                                                                                               
                                temp_map[i, j] = np.sum(element.subelement_weights[:, i, j])
                else:
                    temp_map = np.zeros((self.total_stimuli, self.total_max))
            
                                          
            if self.symbol in ['A', 'B', '+'] and time <= 10:
                print(f"DEBUG {self.symbol}: element temp_map sum: {np.sum(temp_map)}")
                print(f"  element.subelement_weights shape: {element.subelement_weights.shape if hasattr(element, 'subelement_weights') and element.subelement_weights is not None else 'None'}")
                print(f"  element.subelement_weights sum: {np.sum(element.subelement_weights) if hasattr(element, 'subelement_weights') and element.subelement_weights is not None else 'None'}")
                print(f"  total_stimuli: {self.total_stimuli}, total_max: {self.total_max}")
                print(f"  temp_map shape: {temp_map.shape}")
            
                                                                             
            for i in range(self.total_stimuli):
                temp_w = 0.0
                for j in range(self.total_max):
                                                                                                                                           
                                                                                                   
                    if i < len(self.names) and self.names[i] in self.group.get_cues_map():
                                                                                
                        target_stimulus = self.group.get_cues_map().get(self.names[i])
                        divisor = len(target_stimulus.get_list()) if hasattr(target_stimulus, 'get_list') else len(target_stimulus.cues)
                    else:
                        divisor = 1.0
                    
                    temp_w += temp_map[i, j] / divisor
                    
                                                                                  
                self.average_weights_a[i] += temp_w
                                                                        
                self.average_weights[i] = self.average_weights[i] + temp_w
                
                                            
                if self.symbol in ['A', 'B', '+'] and time <= 10 and temp_w > 0:
                    print(f"DEBUG {self.symbol}: temp_w[{i}] = {temp_w}, generalActivation = {element.getGeneralActivation()}")
                    print(f"  average_weights[{i}] = {self.average_weights[i]}")
                    print(f"  average_weights_a[{i}] = {self.average_weights_a[i]}")
                
                                                         
        delta_w = (self.average_weights[self.names.index("+")] - old_v) if "+" in self.names else 0 - old_v
        if self.delta_ws is not None and self.current_phase < len(self.delta_ws) and self.trial_count < len(self.delta_ws[self.current_phase]) and self.timepoint < len(self.delta_ws[self.current_phase][self.trial_count]):
            self.delta_ws[self.current_phase][self.trial_count][self.timepoint] = delta_w
            
        if self.asymptotes is not None and self.current_phase < len(self.asymptotes) and self.trial_count < len(self.asymptotes[self.current_phase]) and self.timepoint < len(self.asymptotes[self.current_phase][self.trial_count]):
            self.asymptotes[self.current_phase][self.trial_count][self.timepoint] = asymptote_mean
            
                                              
        if not self.has_been_active:
            self.duration_point = time - self.last_onset
            
                                                                   
        duration = self.last_offset - self.last_onset if self.has_been_active or self.is_context else self.max_duration
        ctx_fix = 1 if self.is_context else 0
        
                                                  
        if self.average_average_weights is None:
            self.average_average_weights = np.zeros(self.total_stimuli)
        if self.average_average_weights_a is None:
            self.average_average_weights_a = np.zeros(self.total_stimuli)
        
                                                                              
                                                                                   
        if (self.not_divided and (self.is_active() or self.has_been_active) and 
            self.duration_point > 0):
            self.trial_timepoints_counter += 1
            for i in range(self.total_stimuli):
                self.average_average_weights[i] += self.average_weights[i]
                self.average_average_weights_a[i] += self.average_weights_a[i]
            
                                                
            if self.symbol in ['A', 'B', '+'] and time <= 10:
                print(f"DEBUG {self.symbol}: ACCUMULATING weights at time={time}")
                print(f"  average_weights: {self.average_weights}")
                print(f"  average_weights_a: {self.average_weights_a}")
                print(f"  accumulated so far: {self.average_average_weights_a}")
            
                          
            if self.symbol in ['A', 'B', '+'] and time <= 10:                                 
                print(f"DEBUG {self.symbol}: time={time}, active={self.is_active()}, has_been_active={self.has_been_active}, duration_point={self.duration_point}, not_divided={self.not_divided}")
                print(f"  average_weights_a: {self.average_weights_a}")
                print(f"  trial_timepoints_counter: {self.trial_timepoints_counter}")
        
                                                                        
                                                                                     
        if (self.not_divided and self.has_been_active and not self.is_active()):
            for i in range(self.total_stimuli):
                divider = max(1, self.trial_timepoints_counter)                                   
                if divider > 0:
                    self.average_average_weights[i] /= divider
                    self.average_average_weights_a[i] /= divider
            self.not_divided = False
            
                                                                
        elif iti and self.not_divided and self.trial_timepoints_counter > 0:
            for i in range(self.total_stimuli):
                self.average_average_weights[i] /= self.trial_timepoints_counter
                self.average_average_weights_a[i] /= self.trial_timepoints_counter
            self.not_divided = False
            
                                                  
        if iti:
            self.trial_timepoints_counter = 0
            
        self.timepoint += 1
        
        return self.average_weights, self.average_weights_a
        
    def reset_for_next_timepoint(self):
        """Reset elements for the next timepoint."""
        for element in self.cues:
            element.resetForNextTimepoint()
            
    def reset(self, last: bool = False, current_trials: int = 0):
        self.timepoint = 0
        
                                                                    
        if self.cues is not None:
            for cue in self.cues:
                cue.reset(last, current_trials)
        
        self.has_been_active = False
        if not self.is_context:
            self.start_decay = False
        self.presence_trace = 0
        
                                          
        self.average_average_weights = np.zeros(self.total_stimuli)
        
                            
        self.trial_count -= min(self.trial_count, current_trials)
        
        if self.activity is not None and self.current_phase < len(self.activity):
            self.activity[self.current_phase] = np.zeros(self.trials + 1, dtype=bool)
            
        self.duration_point = 0
        self.trial_count = max(0, self.trial_count - current_trials)
        
    def set_phase(self, phase: int):
        """
        Set the current experimental phase.
        
        Args:
            phase: Phase number
        """
        self.current_phase = phase
        
                                               
        if phase == 0:
            if self.was_active_last is not None:
                self.was_active_last = np.zeros_like(self.was_active_last, dtype=bool)
            if self.activity is not None:
                self.activity[0] = np.zeros(self.trials + 1, dtype=bool)
        elif phase > 0:
            if self.was_active_last is not None and phase < len(self.was_active_last):
                self.was_active_last[phase] = self.was_active_last[phase - 1]
                
                               
        self.average_average_weights = np.zeros(self.total_stimuli)
        
                             
        for cue in self.cues:
            cue.setPhase(phase)
            
    def store(self, trial_type: str):
        """
        Store trial data.
        
        Args:
            trial_type: Type of trial
        """
                        
        if self.activity is not None and self.trial_count < len(self.activity[0]):
            self.activity[self.current_phase][self.trial_count] = self.has_been_active
            
                               
        if self.was_active_last is not None and self.current_phase < len(self.was_active_last):
            self.was_active_last[self.current_phase] = self.has_been_active
            
                                                                                    
            
                        
        for cue in self.cues:
            cue.store()
            
        self.trial_count += 1
        
                              
        self.not_divided = True                                     
        self.timepoint = 0
        if not self.is_context:
            self.has_been_active = False
            self.start_decay = False
            self.presence_trace = 0
            
        self.average_average_weights = np.zeros(self.total_stimuli)
        self.average_average_weights_a = np.zeros(self.total_stimuli)
        
                         
    def get_list(self) -> List[StimulusElement]:
        """Get the list of stimulus elements."""
        return self.cues
        
    def get(self, index: int) -> StimulusElement:
        """Get a specific stimulus element."""
        return self.cues[index]
        
    def size(self) -> int:
        """Get the number of stimulus elements."""
        return len(self.cues)
        
    def get_name(self) -> str:
        """Get the stimulus name."""
        return self.symbol
        
    def get_symbol(self) -> str:
        """Get the stimulus symbol."""
        return self.symbol
        
    def is_active(self) -> bool:
        """Check if stimulus is currently active."""
        return self.active and not self.disabled
        
    def get_has_been_active(self) -> bool:
        """Check if stimulus has been active in this trial."""
        return self.has_been_active
        
    def get_trial_count(self) -> int:
        """Get the current trial count."""
        return self.trial_count
        
    def get_alpha(self) -> float:
        """Get the alpha learning rate."""
        return self.alpha_r
        
    def set_alpha_r(self, alpha: float):
        """Set the US learning rate."""
        self.alpha_r = alpha
        for cue in self.cues:
            cue.set_alpha_r(alpha)
            
    def set_alpha_n(self, alpha: float):
        """Set the CS-CS learning rate."""
        self.alpha_n = alpha
        for cue in self.cues:
            cue.set_alpha_n(alpha)
            
    def set_beta(self, beta: float):
        """Set the beta parameter."""
        self.beta = beta
        for cue in self.cues:
            cue.set_beta(beta)
            
    def set_salience(self, salience: float):
        """Set the salience."""
        self.salience = salience
        for cue in self.cues:
            cue.set_salience(salience / self.max_duration if self.max_duration > 0 else salience)
    
    def set_context_reset(self, reset: float):
        """Set context reset value."""
        self.context_reset = reset
    
    def reset_activation(self, reset: bool):
        """Reset activation state."""
        if reset:
            self.presence_trace = 0.0
            self.active = False
            self.has_been_active = False
    
    def initialize(self, a, b):
        if a is None or b is None:
            return

        self.a = a
        self.b = b

        try:
            self.alpha_r = (a.get_alpha() / 2.0) + (b.get_alpha() / 2.0)
        except AttributeError:
            self.alpha_r = (getattr(a, 'alpha_r', self.alpha_r) / 2.0) + (getattr(b, 'alpha_r', self.alpha_r) / 2.0)

        try:
            self.alpha_n = (a.get_alpha_n() / 2.0) + (b.get_alpha_n() / 2.0)
        except AttributeError:
            self.alpha_n = (getattr(a, 'alpha_n', self.alpha_n) / 2.0) + (getattr(b, 'alpha_n', self.alpha_n) / 2.0)

        try:
            sal_a = a.get_salience()
        except AttributeError:
            sal_a = getattr(a, 'salience', 1.0)
        try:
            sal_b = b.get_salience()
        except AttributeError:
            sal_b = getattr(b, 'salience', 1.0)
        self.salience = (sal_a / 2.0) + (sal_b / 2.0)

        for element in self.cues:
            if hasattr(element, 'initialize'):
                element.initialize(a, b)
            if hasattr(element, 'setRAlpha'):
                element.setRAlpha(self.alpha_r)
            if hasattr(element, 'setNAlpha'):
                element.setNAlpha(self.alpha_n)
            if hasattr(element, 'setSalience'):
                element.setSalience(self.salience)
            
    def set_max_duration(self, duration: int):
        """Set the maximum duration."""
        self.max_duration = duration
        
    def set_all_max_duration(self, duration: int):
        """Set the total maximum duration."""
        self.total_max = duration
        
                                          
        if hasattr(self.group, 'get_no_of_phases'):
            num_phases = self.group.get_no_of_phases()
            max_iti = getattr(self.group, 'max_iti', 0)
            
            self.delta_ws = np.zeros((num_phases, self.trials, self.total_max + max_iti))
            self.asymptotes = np.zeros((num_phases, self.trials, self.total_max + max_iti))
            
    def set_disabled(self):
        """Disable this stimulus."""
        self.disabled = True
        self.alpha_r = 0.5
        self.max_duration = 1
        self.last_onset = 0
        self.last_offset = 1
        
    def update_total_stimuli(self, new_total_stimuli: int):
        """
        Update the total number of stimuli for this stimulus and all its elements.
        
        Args:
            new_total_stimuli: New total number of stimuli
        """
        if new_total_stimuli != self.total_stimuli:
            self.total_stimuli = new_total_stimuli
            
                                                              
            if self.total_max == 0:
                self.total_max = 100                           
            
                                                         
            self._initialize_arrays()

                                                               
                                                                    
            self.trial_w = None
            self.trial_wa = None
            self.initialize_trial_arrays()

                                 
            for element in self.cues:
                element.update_total_stimuli(new_total_stimuli)
        
    def is_common(self) -> bool:
        """Check if this is a common/compound stimulus."""
        return len(self.symbol) > 1 and self.symbol[0] == 'c'
    
    def get_was_active(self) -> bool:
        """Get was active status."""
        return getattr(self, 'was_active', False)
    
    def get_last_onset(self) -> int:
        """Get the last onset time."""
        return self.last_onset
    
    def get_should_update(self) -> bool:
        """Get whether this stimulus should update."""
        return getattr(self, 'update', True)
    
    def get_salience(self) -> float:
        """Get the salience value."""
        return self.salience
    
    def get_names(self) -> List[str]:
        """Get the names list."""
        return self.names
    
    def post_store(self):
        """Post-store processing."""
                                    
        pass
        
    def set_reset_context(self, reset_context: bool):
        """Set context reset flag."""
        self.reset_context = reset_context
        
    def set_zero_probe(self):
        """Set zero probe flag."""
        self.is_probe = False
    
    def get_b(self) -> float:
        """Get b parameter."""
        return getattr(self, 'b', 1.0)
    
    def get_common_map(self) -> Dict[str, 'Stimulus']:
        """Get common map."""
        return getattr(self, 'common_map', {})
    
    def get_prediction(self, name: str) -> float:
        """Get prediction for a stimulus name."""
        if name in self.names and self.average_average_weights_a is not None:
                                                                                  
            prediction = self.average_average_weights_a[self.names.index(name)]
            return prediction
        return 0.0
    
    def get_v_value(self, name: str) -> float:
        """Get V value for a stimulus name."""
        if name in self.names and self.average_weights is not None:
                                                                  
            v_value = self.average_weights[self.names.index(name)]
                                                                                       
            return v_value
        return 0.0
    
    def prestore(self):                                                                  
        if self.trial_w is not None and self.trial_count < len(self.trial_w[0][0]):
                                                             
                                                                                                 
            self.trial_w[0][self.current_phase][self.trial_count] = self.average_average_weights.copy()
            self.trial_wa[0][self.current_phase][self.trial_count] = self.average_average_weights_a.copy()
    
    def get_trial_average_weights(self, trial_type: int, phase: int) -> np.ndarray:
        """
        Get trial average weights for a specific trial type and phase.
    Equivalent method: getTrialAverageWeights(int trialType, int phase)

    This returns the final accumulated weights (averageAverageWeights)
    that are stored in trialW arrays - this is what the GUI displays.
        
        Args:
            trial_type: Trial type index
            phase: Phase index
            
        Returns:
            Array of final accumulated weights for all stimuli
        """
        if self.trial_w is not None and trial_type < len(self.trial_w) and phase < len(self.trial_w[0]):
                                                           
            trial_data = self.trial_w[trial_type][phase]
            if len(trial_data) > 0:
                return trial_data[-1]              
            return np.zeros(self.total_stimuli)
        return np.zeros(self.total_stimuli)
    
    def get_trial_average_weights_a(self, trial_type: int, phase: int) -> np.ndarray:
        """
        Get trial average weights A for a specific trial type and phase.
    Equivalent method: getTrialAverageWeightsA(int trialType, int phase)

    This returns the final accumulated activation-scaled weights (averageAverageWeightsA)
    that are stored in trialWA arrays.
        
        Args:
            trial_type: Trial type index
            phase: Phase index
            
        Returns:
            Array of final accumulated activation-scaled weights for all stimuli
        """
        if self.trial_wa is not None and trial_type < len(self.trial_wa) and phase < len(self.trial_wa[0]):
                                                           
            trial_data = self.trial_wa[trial_type][phase]
            if len(trial_data) > 0:
                return trial_data[-1]              
            return np.zeros(self.total_stimuli)
        return np.zeros(self.total_stimuli)

    def get_all_max_duration(self) -> int:
        """Get the total max duration."""
        return self.total_max
