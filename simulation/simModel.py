import numpy as np
from typing import Dict, List, Optional, Any, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import pickle
from pathlib import Path
import time
import threading
from collections import OrderedDict
import copy

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.sim_group import SimGroup
from simulation.model_control import ModelControl


class SimModel:
    def __init__(self):
                                
        self.groups: "OrderedDict[str, SimGroup]" = OrderedDict()
        self.groups_no = 1
        self.phases_no = 1
        self.combination_no = 4                                      
        self.variable_distribution_combination_no = 4
        
                                                            
        self.config_cues_names: Dict[str, str] = {}                
        self.config_cue_counter: int = 0
        
                                    
        self.random_seed = int(time.time() * 1000) % (2**31)                                   
        
                       
        self.root_directory: Optional[Path] = None
        
                                                       
        self.timestep_size = 1.0                      
        self.list_all_cues: List[str] = []                            
        self.threshold = 0.875                
        self.use_context = True                      
        
                                          
        self.show_response = True
        self.is_context_across_phase = False
        self.is_csc = True                
        self.is_geo = False
        self.is_exponential = True                
        self.context_alpha_r = 0.25                         
        self.context_alpha_n = 0.2                              
        self.context_salience = 0.07                            
        
                                                                                     
        self.alpha_r = 0.5                                 
        self.alpha_n = 0.5                                 
        
                                               
        self.decision_rule = "CHURCH_KIRKPATRICK"                               
        self.random_phases = 0
        self.decay = 0.5                
        self.timing_per_trial = False
        self.is_external_save = True
        
                                   
        self.serial_configurals = True                
        self.is_zero_trace = False
        self.serial_compounds = False
        self.restrict_predictions = True                
        
                               
        self.activation_decay = 0.15
        self.activation_dropoff = 0.2
        
                                             
        self.skew = 20.0
        self.reset_value = 0.95
        self.us_cv = 20.0
        self.is_compounds = False                                               
        self.persistence = 0
        self.set_size = 10
        self.cs_scalar = 2.5
        self.cs_cv = 20.0
        self.rule = 2
        self.discount = 0.999                
        self.buffer_size = 1.0
        self.associative_discount = 0.999                
        
                             
        self.responses_per_minute = 100                
        self.reset_context = True
        self.serial_response_weight = 0.85
        self.is_configural_compounds = False
        
                        
        self._is_errors = False
        self._is_errors2 = False
        
                                                      
        self.intensities: Dict[str, List[float]] = {}                                     
        self.common_map: Dict[str, Dict[str, float]] = {}                                           
        self.proportions: Dict[str, Dict[str, float]] = {}                                           
        self.max_phase = 1
        
                                   
        self.stim_names: List[str] = []
        self.stim_names2: List[str] = []
        
                                     
        self.control: Optional[ModelControl] = None
        self.view: Optional[Any] = None                     

                                
        self._initialize_collections()
        
    def _initialize_collections(self):
        """Initialize collections that need setup."""
        if not self.intensities:
            self.intensities = {}
        if not self.common_map:
            self.common_map = {}
            
                                                                     
        if self.use_context:
            omega_symbol = "Ω"                           
            if omega_symbol not in self.list_all_cues:
                self.list_all_cues.append(omega_symbol)
    
    def add_cue_names(self, cue_map: Dict[str, Any]):
        for name, stimulus in cue_map.items():
            symbol = stimulus.get_symbol() if hasattr(stimulus, 'get_symbol') else name
            self.values[symbol] = 0.0                 
            
                                                                
            if symbol not in self.list_all_cues:
                self.list_all_cues.append(symbol)
        
                                          
        self.list_all_cues.sort()
    
    def add_group_into_map(self, name: str, group: SimGroup):
        self.groups[name] = group
        self.add_cue_names(group.get_cues_map())
        self.random_phases += group.num_random() if hasattr(group, 'num_random') else 0
    
    def add_phase_to_sessions(self):
        """Add a new phase to group sessions tracking."""
        self.group_sessions.append({})
    
    def add_session_to_group(self, phase: int, group_name: str, session: int):
        """
        Add session information for a group in a specific phase.
        
        Args:
            phase: Phase number
            group_name: Name of the group
            session: Session number
        """
        if phase < len(self.group_sessions):
            self.group_sessions[phase][group_name] = session
    
    def get_group_session(self, group_name: str, phase: int) -> int:
        """Get session number for a group in a phase."""
        return 1                             
    
    def add_values_into_map(self):                       
        for p in range(1, self.phases_no + 1):
            at_least_one_group_plus = False
            at_least_one_group_minus = False
            
                                                               
            for group in self.groups.values():
                phases = group.get_phases()
                if len(phases) >= p:
                    phase = phases[p - 1]
                    if hasattr(phase, 'get_lambda_plus') and phase.get_lambda_plus() is not None:
                        at_least_one_group_plus = True
                    if hasattr(phase, 'get_lambda_minus') and phase.get_lambda_minus() is not None:
                        at_least_one_group_minus = True
                    if at_least_one_group_plus and at_least_one_group_minus:
                        break
            
                                            
            if at_least_one_group_plus:
                self.values[f"lambda p{p}"] = None
                self.values[f"alpha+ p{p}"] = None
            if at_least_one_group_minus:
                self.values[f"lambda p{p}"] = None
                self.values[f"alpha- p{p}"] = None
    
    def set_is_errors(self, b: bool, b2: bool):
        """Set error tracking flags."""
        self._is_errors = b
        self._is_errors2 = b2

    def is_errors(self) -> bool:
        """Check if errors are tracked."""
        return self._is_errors

    def is_errors2(self) -> bool:
        """Check if secondary errors are tracked."""
        return self._is_errors2
    
    def clear_configural_map(self):
        """Clear configural cue mappings."""
        self.config_cues_names.clear()
    
    def context_across_phase(self) -> bool:
        """Check if context spans across phases."""
        return self.is_context_across_phase
    
    def cue_name_2_interface_name(self, cue_name: str) -> str:
        configural = False
        configurals = ""
        
                                                               
        for char in cue_name:
            if 'a' <= char <= 'z' and char != 'ω':                 
                configural = True
                configurals += char
        
        if configural and cue_name != "Ω":
            if len(cue_name) == 1:
                                                         
                compound_name = self.config_cues_names.get(cue_name, "")
                interface_name = f"c({compound_name})"
            elif "'" in cue_name:
                                                     
                compound_name = cue_name.replace(configurals, "")
                interface_name = compound_name
            else:
                                                               
                compound_name = cue_name
                for conf in configurals:
                    compound_name = compound_name.replace(conf, "")
                interface_name = f"[{compound_name}]"
        else:
            interface_name = cue_name
            
        return interface_name
    
    def interface_name_2_cue_name(self, interface_name: str) -> str:
        if "(" in interface_name:                              
            compound = interface_name[2:-1]                         
            cue_name = self._get_key_by_value(self.config_cues_names, compound)
        elif "[" in interface_name:                       
            compound = interface_name[1:-1]                        
            key = self._get_key_by_value(self.config_cues_names, compound)
            cue_name = compound + (key if key else "")
        else:
            cue_name = interface_name
            
        return cue_name if cue_name else interface_name
    
    def _get_key_by_value(self, mapping: Dict[str, str], value: str) -> Optional[str]:
        """Find key for a given value in mapping."""
        for key, val in mapping.items():
            if val == value:
                return key
        return None
    
    def get_alpha_cues(self) -> Dict[str, float]:
        alpha_cues = {}
        for name, value in self.values.items():
            if "lambda" not in name and "alpha" not in name:
                alpha_cues[name] = value
        return alpha_cues
    
    def get_number_alpha_cues(self) -> int:
        count = 0
        for name in self.values.keys():
            if "lambda" in name or "alpha" in name:
                count += 1
        return len(self.values) - count
    
    def get_contexts(self) -> Dict[str, Any]:
        contexts = {}
        for group in self.groups.values():
            if hasattr(group, 'get_contexts'):
                contexts.update(group.get_contexts())
        return contexts
    
    def reinitialize(self):
        self.values = {}
        self.groups = OrderedDict()
        self.config_cues_names = {}
                                                                      
    
    def start_calculations(self):                    
        self.list_all_cues.clear()
        
                                                                          
        with ThreadPoolExecutor(max_workers=len(self.groups)) as executor:
                                              
            future_to_group = {}
            for group in self.groups.values():
                future = executor.submit(self._process_group_with_cue_collection, group)
                future_to_group[future] = group
            
                                              
            for future in as_completed(future_to_group):
                group = future_to_group[future]
                try:
                    cue_names = future.result()
                                                   
                    for name in cue_names:
                        if name not in self.list_all_cues:
                            self.list_all_cues.append(name)
                except Exception as e:
                    print(f"Error processing group {group.get_name() if hasattr(group, 'get_name') else 'unknown'}: {e}")
            
                             
            if self.control:
                self.control.increment_progress(1)
        
                       
        self.list_all_cues.sort()
        
                       
        if self.control:
            self.control.set_complete(True)
    
    def _process_group_with_cue_collection(self, group: SimGroup) -> List[str]:
                     
        group.run()
        
                           
        cue_names = []
        cues_map = group.get_cues_map()
        for stimulus in cues_map.values():
            name = stimulus.get_symbol() if hasattr(stimulus, 'get_symbol') else stimulus.get_name()
            if name not in cue_names:
                cue_names.append(name)
        
        return cue_names
    
    def total_num_phases(self) -> int:
        total = 0
        for group in self.groups.values():
            if hasattr(group, 'trial_count'):
                total += group.trial_count()
        return total
    
    def update_values(self, name: str, phase: int, value: str):                                           
        us_names = []
        alpha_plus_names = []
        beta_names = []
        omicron_names = []
        us_lambda_names = []
        
        for cue_name in self.list_all_cues:
            if self._is_us(cue_name) and cue_name not in us_names:
                us_names.append(cue_name)
                alpha_plus_names.append(f"{cue_name} - alpha+")
                us_lambda_names.append(f"{cue_name} - lambda")
                beta_names.append(f"{cue_name} - β")
                omicron_names.append(f"{cue_name}_s")
        
        if value == "":
                                                             
            self._handle_empty_value_update(name, phase, alpha_plus_names, beta_names, 
                                           omicron_names, us_lambda_names)
        else:
                                     
            if False:                                                       
                compound_name = name[2:-1]                            
                virtual_name = self._get_key_by_value(self.config_cues_names, compound_name)
                self.values[virtual_name] = float(value)
            else:
                self.values[name] = float(value)
                self.values[f"{name} p{phase}"] = float(value)
    
    def _handle_empty_value_update(self, name: str, phase: int, alpha_plus_names: List[str],
                                  beta_names: List[str], omicron_names: List[str], 
                                  us_lambda_names: List[str]):
        """Handle parameter updates when value is empty."""
                                 
        is_alpha_plus = False
        for alpha_plus in alpha_plus_names:
            if alpha_plus in name:
                default_key = f"{alpha_plus} p1"
                if default_key in self.values:
                    self.values[f"{name} p{phase}"] = self.values[default_key]
                else:
                    self.values[f"{name} p{phase}"] = float(value) if value else 0.0
                is_alpha_plus = True
                break
        
        if not is_alpha_plus and "alpha+" in name:
            default_key = "alpha+ p1"
            if default_key in self.values:
                self.values[f"{name} p{phase}"] = self.values[default_key]
            else:
                self.values[f"{name} p{phase}"] = 0.0
            is_alpha_plus = True
        
                                  
        is_omicron = False
        for omicron in omicron_names:
            if omicron in name:
                default_key = f"{omicron} p1"
                if default_key in self.values:
                    self.values[f"{name} p{phase}"] = self.values[default_key]
                is_omicron = True
                break
        
        if not is_omicron and "+_s" in name:
            default_key = "+_s p1"
            if default_key in self.values:
                self.values[f"{name} p{phase}"] = self.values[default_key]
            is_omicron = True
        
                               
        is_beta = False
        for beta in beta_names:
            if beta in name:
                default_key = f"{beta} p1"
                if default_key in self.values:
                    self.values[f"{name} p{phase}"] = self.values[default_key]
                is_beta = True
                break
        
        if not is_beta and "β" in name:
            default_key = "β p1"
            if default_key in self.values:
                self.values[f"{name} p{phase}"] = self.values[default_key]
            is_beta = True
        
                                 
        is_lambda = False
        for lambda_name in us_lambda_names:
            if lambda_name in name:
                default_key = f"{lambda_name} p1"
                if default_key in self.values:
                    self.values[f"{name} p{phase}"] = self.values[default_key]
                is_lambda = True
                break
        
        if not is_lambda and "lambda" in name:
            default_key = "lambda p1"
            if default_key in self.values:
                self.values[f"{name} p{phase}"] = self.values[default_key]
            is_lambda = True
        
                                                                   
        if not (is_lambda or is_beta or is_omicron or is_alpha_plus):
            self._handle_other_parameters(name, phase)
    
    def _handle_other_parameters(self, name: str, phase: int):
        """Handle non-US parameter updates."""
        parameter_mappings = {
            "reinforcer cff": "reinforcer cff p1",
            "integration": "integration p1", 
            "US ρ": "US ρ p1",
            "Threshold": "Threshold p1",
            "gamma": "gamma p1",
            "Variable Salience": "Variable Salience p1",
            "skew": "skew p1",
            "φ": "φ p1",
            "Wave Constant": "Wave Constant p1",
            "US Scalar Constant": "US Scalar Constant p1",
            "delta": "delta p1",
            "b": "b p1",
            "common": "common p1",
            "setsize": "setsize p1",
            "ς": "ς p1",
            "CV": "CV p1",
            "linear c": "linear c p1",
            "τ1": "τ1 p1",
            "τ2": "τ2 p1",
            "Salience Weight": "Salience Weight p1",
            "ϑ": "ϑ p1",
            "CS ρ": "CS ρ p1",
            "Self Discount": "Self Discount p1",
            "ω": "ω p1"
        }
        
        for param, default_key in parameter_mappings.items():
            if param in name:
                if default_key in self.values:
                    self.values[f"{name} p{phase}"] = self.values[default_key]
                break
    
    def initialize_intensities(self):
        for group in self.groups.values():
            if hasattr(group, 'get_no_of_phases'):
                self.max_phase = max(self.max_phase, group.get_no_of_phases())
        
                                               
        for group_name in self.groups.keys():
            if group_name not in self.intensities:
                self.intensities[group_name] = []
            
            group = self.groups[group_name]
            group_phases = group.get_no_of_phases() if hasattr(group, 'get_no_of_phases') else 1
            
            for i in range(1, self.max_phase + 1):
                if i <= group_phases and len(self.intensities[group_name]) <= i:
                    self.intensities[group_name].append(1.0)                     
    
    def set_intensities(self, intensities_map: Dict[str, List[float]]):
        for group_name, group in self.groups.items():
            phases = group.get_phases() if hasattr(group, 'get_phases') else []
            for i, phase in enumerate(phases):
                intensity = 1.0           
                if (intensities_map and group_name in intensities_map and 
                    i < len(intensities_map[group_name])):
                    intensity = intensities_map[group_name][i]
                
                if hasattr(phase, 'set_intensity'):
                    phase.set_intensity(intensity)
    
    def initialize_common_map(self):
        if not self.common_map:
            common_value = 0.0                                               
            if hasattr(self, 'view') and self.view:
                try:
                                                                              
                    common_value = 2.0                                 
                except:
                    common_value = 2.0
            
            for group in self.groups.values():
                group_name = group.get_name() if hasattr(group, 'get_name') else str(group)
                self.stim_names = []
                self.common_map[group_name] = {}
                
                cues_map = group.get_cues_map() if hasattr(group, 'get_cues_map') else {}
                for stimulus in cues_map.values():
                    stim_name = stimulus.get_name() if hasattr(stimulus, 'get_name') else str(stimulus)
                    is_common = stimulus.is_common() if hasattr(stimulus, 'is_common') else False
                    
                    if stim_name not in self.stim_names and is_common:
                        self.stim_names.append(stim_name)
                        self.common_map[group_name][stim_name] = common_value
    
    def set_common_map(self, common_map: Dict[str, Dict[str, float]]):
        if common_map is not None:
            self.common_map = common_map
        self.calculate_common_proportions()
    
    def calculate_common_proportions(self):
        self.proportions = {}
        
                                               
        for group_name in self.groups.keys():
            self.proportions[group_name] = {}
        
                               
        for group_name, group in self.groups.items():
            cues_map = group.get_cues_map() if hasattr(group, 'get_cues_map') else {}
            for stim_name in cues_map.keys():
                total = 0.0
                if group_name in self.common_map:
                    for common_name in self.common_map[group_name].keys():
                        if stim_name in common_name:
                            total += self.common_map[group_name][common_name]
                self.proportions[group_name][stim_name] = total
        
                                       
        for group_name in self.proportions.keys():
            for stim_name in self.proportions[group_name].keys():
                if self.proportions[group_name][stim_name] >= 1:
                    for common_name in self.stim_names:
                        if stim_name in common_name and group_name in self.common_map:
                            if common_name in self.common_map[group_name]:
                                current_val = self.common_map[group_name][common_name]
                                proportion = self.proportions[group_name][stim_name]
                                self.common_map[group_name][common_name] = current_val / proportion
    
    def update_values_on_groups(self):
        for group_name, group in self.groups.items():
                                    
            if hasattr(group, 'clear_results'):
                group.clear_results()
            
                                        
            cues_map = group.get_cues_map() if hasattr(group, 'get_cues_map') else {}
            for stimulus in cues_map.values():
                self._update_stimulus_parameters(stimulus)
            
                                          
            self._update_us_parameters(group)
            
                                                    
            for p in range(1, self.phases_no + 1):
                self._update_phase_parameters(group, p)
    
    def _update_stimulus_parameters(self, stimulus: Any):
        """Update individual stimulus parameters."""
        if not hasattr(stimulus, 'get_name'):
            return
            
        name = stimulus.get_name()
        
                        
        if hasattr(stimulus, 'reset'):
            stimulus.reset(False, 0)
        
                              
        alpha_r_key = f"{name}_αr"
        alpha_n_key = f"{name}_αn"
        salience_key = f"{name}_s"
        
        alpha_r_value = self.values.get(alpha_r_key, -1.0)
        alpha_n_value = self.values.get(alpha_n_key, -1.0)
        salience = self.values.get(salience_key, -1.0)
        
                                                                        
        if len(name) > 1:
            alpha_r_sum = 0.0
            alpha_n_sum = 0.0
            char_count = 0
            
            for char in name:
                if char != 'c':                             
                    if char in self.values:
                        alpha_r_sum += self.values.get(char, 0.0)
                        alpha_n_sum += self.values.get(char, 0.0)
                        char_count += 1
            
            if char_count > 0:
                alpha_r_value = alpha_r_sum / char_count
                alpha_n_value = alpha_n_sum / char_count
        
                                 
        if alpha_r_value != -1.0 and alpha_r_value != 0:
            if hasattr(stimulus, 'set_r_alpha') or hasattr(stimulus, 'set_alpha_r'):
                setter = getattr(stimulus, 'set_r_alpha', getattr(stimulus, 'set_alpha_r', None))
                if setter:
                    setter(alpha_r_value)
        
        if alpha_n_value != -1.0 and alpha_n_value != 0:
            if hasattr(stimulus, 'set_n_alpha') or hasattr(stimulus, 'set_alpha_n'):
                setter = getattr(stimulus, 'set_n_alpha', getattr(stimulus, 'set_alpha_n', None))
                if setter:
                    setter(alpha_n_value)
        
        if salience != -1.0 and salience != 0:
            if hasattr(stimulus, 'set_salience'):
                stimulus.set_salience(salience)
    
    def _update_us_parameters(self, group: SimGroup):
        """Update US (unconditioned stimulus) parameters for a group."""
        us_names = []
        us_alpha_names = []
        us_beta_names = []
        us_omicron_names = []
        
                                   
        for cue_name in self.list_all_cues:
            if self._is_us(cue_name) and cue_name not in us_names:
                us_names.append(cue_name)
                us_alpha_names.append(f"{cue_name} - α+")
                us_beta_names.append(f"{cue_name} - β")
                us_omicron_names.append(f"{cue_name}_s")
        
        cues_map = group.get_cues_map() if hasattr(group, 'get_cues_map') else {}
        
                                        
        for alpha_name in us_alpha_names:
            us_name = alpha_name.split(" - ")[0]
            key = alpha_name if len(us_alpha_names) > 1 else "alpha+ p1"
            if key in self.values and us_name in cues_map:
                if hasattr(cues_map[us_name], 'set_n_alpha'):
                    cues_map[us_name].set_n_alpha(self.values[key])
        
                                          
        for omicron_name in us_omicron_names:
            us_name = omicron_name.split("_")[0]
            key = omicron_name if len(us_omicron_names) > 1 else "+_s"
            if key in self.values and us_name in cues_map:
                if hasattr(cues_map[us_name], 'set_omicron'):
                    cues_map[us_name].set_omicron(self.values[key])
        
                                       
        for beta_name in us_beta_names:
            us_name = beta_name.split(" - ")[0]
            key = beta_name if len(us_beta_names) > 1 else "β"
            if key in self.values and us_name in cues_map:
                if hasattr(cues_map[us_name], 'set_beta'):
                    cues_map[us_name].set_beta(self.values[key])
    
    def _update_phase_parameters(self, group: SimGroup, phase_num: int):
        """Update parameters for a specific phase."""
        phases = group.get_phases() if hasattr(group, 'get_phases') else []
        if phase_num > len(phases):
            return
        
        phase = phases[phase_num - 1]
        p = phase_num
        
                  
        if hasattr(group, 'set_rule'):
            group.set_rule(self.rule)
        
                      
        key = f"gamma p{p}"
        if key in self.values and hasattr(phase, 'set_gamma'):
            phase.set_gamma(self.values[key])
        
                       
        key = f"common p{p}"
        if key in self.values and hasattr(group, 'set_common'):
            group.set_common(self.values[key])
        
                                          
        key = f"Wave Constant p{p}"
        if key in self.values and hasattr(phase, 'set_cs_scalar'):
            phase.set_cs_scalar(self.values[key])
        
                            
        key = f"b p{p}"
        if key in self.values:
            cues_map = group.get_cues_map() if hasattr(group, 'get_cues_map') else {}
            for stimulus in cues_map.values():
                if hasattr(stimulus, 'set_b'):
                    stimulus.set_b(self.values[key])
        
                             
        key = f"ϑ p{p}"
        if key in self.values and hasattr(phase, 'set_vartheta'):
            phase.set_vartheta(self.values[key])
        
                                     
        us_rho_key = f"US ρ p{p}"
        cs_rho_key = f"CS ρ p{p}"
        if us_rho_key in self.values and cs_rho_key in self.values:
            if hasattr(phase, 'set_leak'):
                phase.set_leak(self.values[us_rho_key], self.values[cs_rho_key])
        
                      
        key = f"delta p{p}"
        if key in self.values and hasattr(phase, 'set_delta'):
            phase.set_delta(self.values[key])
        
                              
        key = f"Self Discount p{p}"
        if key in self.values and hasattr(phase, 'set_self_prediction'):
            phase.set_self_prediction(self.values[key])
        
                                     
        key = f"ω p{p}"
        if key in self.values and hasattr(phase, 'set_context_salience'):
            phase.set_context_salience(self.values[key])
        
                           
        if hasattr(phase, 'set_reset_context'):
            phase.set_reset_context(self.reset_context)
        
                                 
        if hasattr(phase, 'set_std'):
            phase.set_std(self.get_cs_cv())
        if hasattr(phase, 'set_us_std'):
            phase.set_us_std(self.get_us_cv())
        
                         
        if hasattr(phase, 'set_us_persistence'):
            phase.set_us_persistence(self.get_persistence())
        
                                
        if hasattr(phase, 'set_csc_like'):
            phase.set_csc_like(max(0, self.get_skew(False)))
        
                                 
        if hasattr(phase, 'set_context_reset'):
            phase.set_context_reset(self.get_reset_value(False))
        
                         
        if hasattr(phase, 'set_subset_size'):
            phase.set_subset_size(int(max(1, round(self.get_set_size()))))
    
    def _is_us(self, name: str) -> bool:
        """Check if a stimulus name represents a US."""
                                                                      
        return name == "+" or "US" in name or name.endswith("+")
    
                                   
    def get_activation_decay(self) -> float:
        return self.activation_decay
    
    def get_activation_dropoff(self) -> float:
        return self.activation_dropoff
    
    def get_combination_no(self) -> int:
        return self.combination_no
    
    def get_config_cues_names(self) -> Dict[str, str]:
        return self.config_cues_names
    
    def get_context_alpha(self) -> float:
        return self.context_alpha_r
    
    def get_cue_names(self) -> Set[str]:
        return set(self.values.keys())
    
    def get_decay(self) -> float:
        return self.decay
    
    def get_decision_rule(self) -> str:
        return self.decision_rule
    
    def get_directory(self) -> Optional[Path]:
        return self.root_directory
    
    def get_group_no(self) -> int:
        return self.groups_no
    
    def get_groups(self) -> Dict[str, SimGroup]:
        return self.groups
    
    def get_list_all_cues(self) -> List[str]:
        return self.list_all_cues
    
    def get_phase_no(self) -> int:
        return self.phases_no
    
    def get_random_seed(self) -> int:
        return self.random_seed
    
    def get_threshold(self) -> float:
        return self.threshold
    
    def get_timestep_size(self) -> float:
        return self.timestep_size
    
    def get_values(self) -> Dict[str, float]:
        return self.values
    
    def get_variable_combination_no(self) -> int:
        return self.variable_distribution_combination_no
    
    def get_skew(self, gui: bool) -> float:
        return self.skew
    
    def get_reset_value(self, gui: bool) -> float:
        return self.reset_value
    
    def get_us_cv(self) -> float:
        return self.us_cv
    
    def get_discount(self) -> float:
        return self.discount
    
    def get_responses_per_minute(self) -> int:
        return self.responses_per_minute
    
    def get_serial_response_weight(self) -> float:
        return self.serial_response_weight
    
    def get_associative_discount(self) -> float:
        return self.associative_discount
    
    def get_buffer_size(self) -> float:
        return self.buffer_size
    
    def get_set_size(self) -> int:
        return self.set_size
    
    def get_cs_scalar(self) -> float:
        return self.cs_scalar
    
    def get_persistence(self) -> float:
        return self.persistence
    
    def get_cs_cv(self) -> float:
        return self.cs_cv
    
    def get_common(self) -> Dict[str, Dict[str, float]]:
        return self.common_map
    
    def get_intensities(self) -> Dict[str, List[float]]:
        return self.intensities
    
    def get_max_phase(self) -> int:
        return self.max_phase
    
    def get_proportions(self) -> Dict[str, Dict[str, float]]:
        return self.proportions
    
    def get_context_across_phase(self) -> bool:
        return self.is_context_across_phase
    
                                  
    def is_csc(self) -> bool:
        return self.is_csc
    
    def is_exponential(self) -> bool:
        return self.is_exponential
    
    def is_geometric_mean(self) -> bool:
        return self.is_geo
    
    def is_restrict_predictions(self) -> bool:
        return self.restrict_predictions
    
    def is_serial_compounds(self) -> bool:
        return self.serial_compounds
    
    def is_serial_configurals(self) -> bool:
        return self.serial_configurals
    
    def is_timing_per_trial(self) -> bool:
        return self.timing_per_trial
    
    def is_external_save(self) -> bool:
        return self.is_external_save
    
    def is_use_context(self) -> bool:
        return self.use_context
    
    def is_zero_traces(self) -> bool:
        return self.is_zero_trace
    
    def show_response(self) -> bool:
        return self.show_response
    
    def is_configural_compounds(self) -> bool:
        return self.is_configural_compounds
    
    def is_compound(self) -> bool:
        return self.is_compounds
    
                                   
    def set_activation_decay(self, decay: float):
        self.activation_decay = decay
    
    def set_activation_dropoff(self, dropoff: float):
        self.activation_dropoff = dropoff
    
    def set_combination_no(self, r: int):
        self.combination_no = r
    
    def set_context_across_phase(self, on: bool):
        self.is_context_across_phase = on
    
    def set_context_alpha_r(self, alpha: float):
        self.context_alpha_r = alpha
    
    def set_context_alpha_n(self, alpha: float):
        self.context_alpha_n = alpha
    
    def set_context_salience(self, salience: float):
        self.context_salience = salience
    
    def set_control(self, control: ModelControl):
        self.control = control
        for group in self.groups.values():
            if hasattr(group, 'set_control'):
                group.set_control(control)
    
    def set_csc(self, on: bool):
        self.is_csc = on
    
    def set_decay(self, decay: float):
        self.decay = decay
    
    def set_decision_rule(self, rule: str):
        self.decision_rule = rule
    
    def set_group_no(self, g: int):
        self.groups_no = g
    
    def set_is_exponential(self, exp: bool):
        self.is_exponential = exp
    
    def set_is_geometric_mean(self, on: bool):
        self.is_geo = on
    
    def set_phase_no(self, p: int):
        self.phases_no = p
    
    def set_random_seed(self, seed: int):
        self.random_seed = seed
    
    def set_restrict_predictions(self, restrict: bool):
        self.restrict_predictions = restrict
    
    def set_serial_compounds(self, on: bool):
        self.serial_compounds = on
    
    def set_serial_configurals(self, on: bool):
        self.serial_configurals = on
    
    def set_show_response(self, on: bool):
        self.show_response = on
    
    def set_threshold(self, n: float):
        self.threshold = n
    
    def set_timestep_size(self, size: float):
        self.timestep_size = size
    
    def set_timing_per_trial(self, timing: bool):
        self.timing_per_trial = timing
    
    def set_external_save(self, save: bool):
        self.is_external_save = save
    
    def set_use_context(self, on: bool):
        self.use_context = on
        omega_symbol = "Ω"
        if on:
            if omega_symbol not in self.list_all_cues:
                self.list_all_cues.append(omega_symbol)
        else:
            if omega_symbol in self.list_all_cues:
                self.list_all_cues.remove(omega_symbol)
    
    def set_variable_combination_no(self, num: int):
        self.variable_distribution_combination_no = num
    
    def set_zero_traces(self, on: bool):
        self.is_zero_trace = on

    
    def set_directory(self, directory: Path):
        self.root_directory = directory
    
    def set_reset_context(self, r: bool):
        self.reset_context = r
    
    def set_configural_compounds(self, compounds: bool):
        self.is_configural_compounds = compounds
    
    def set_is_compound(self, b: bool):
        self.is_compounds = b
    
    def set_serial_response_weight(self, weight: float):
        self.serial_response_weight = weight
    
    def set_responses_per_minute(self, rpm: int):
        self.responses_per_minute = rpm
    
    def set_reset_value(self, s: float):
        self.reset_value = s
    
    def set_skew(self, s: float):
        self.skew = s
    
    def set_us_cv(self, cv: float):
        self.us_cv = cv
    
    def set_discount(self, d: float):
        self.discount = d
    
    def set_associative_discount(self, d: float):
        self.associative_discount = d
    
    def set_persistence(self, n: int):
        self.persistence = n
    
    def set_set_size(self, n: int):
        self.set_size = n
    
    def set_cs_scalar(self, n: float):
        self.cs_scalar = n
    
    def set_cs_cv(self, n: float):
        self.cs_cv = n
    
    def set_learning_rule(self, i: int):
        self.rule = i
    
    def set_view(self, view: Any):
        self.view = view
    
                                       
    def run(self):
        """Run method for threading - calls start_calculations."""
        self.start_calculations()
    
                                      
    def add_group(self, name: str) -> SimGroup:
        group = SimGroup(name, self)
        self.add_group_into_map(name, group)
        return group
    
    def set_phases(self, num_phases: int):
        self.set_phase_no(num_phases)
    
    def set_combinations(self, num_combinations: int):
        self.set_combination_no(num_combinations)
    def initialize(self):
        for group in self.groups.values():
            if hasattr(group, "initialize"):
                group.initialize()
                                                        
        self.list_all_cues = []
        for group in self.groups.values():
            if hasattr(group, "cues_map"):
                self.list_all_cues.extend(list(group.cues_map.keys()))
                           
        self.list_all_cues = sorted(set(self.list_all_cues))
        