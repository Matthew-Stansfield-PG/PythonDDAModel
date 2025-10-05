import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional
import random
from core.SimElement import StimulusElement


class SimPhase:
    def __init__(self, phase_num: int, sessions: int, seq: str, order: List,
                 stimuli2: Dict, sg, random_order: bool, timing, iti, context,
                 trials_in_all_phases: int, listed_stimuli: List[str],
                 varying_vartheta: bool):
                          
        self.phaseNum = phase_num
        self.sessions = sessions
        self.initialSeq = seq
        self.stimuli = stimuli2
        self.orderedSeq = order
        self.group = sg
        self.random = random_order
        self.randomize = False
        self.trials = len(order)
        self.trialsInAllPhases = trials_in_all_phases + 1
        self.listedStimuli = listed_stimuli
        self.varyingVartheta = varying_vartheta

                                
        self.timingConfig = timing
        self.itis = iti
        self.contextCfg = context

                                                        
        self.csScalar = 1.0
        self.usScalar = 1.0
        self.usPersistence = 1.0
        self.std = 1.0
        self.contextReset = 0.95
        self.resetContext = False
        self.intensity = 1.0
        self.integration = 0.2
        self.csIntegration = 0.2
        self.leak = 0.99
        self.csLeak = 0.99
        self.dopamine = 1.0
        self.setsize = 100
        self.usCV = 0.0
        self.selfDiscount = 1.0
        self.proportion = 0.0
        self.timepoints = 0
        self.vartheta = 0.5
        self.delta = None
        self.gamma = None
        self.lambdaPlus = None
        self.lambdaMinus = None
        self.betaPlus = None
        self.betaMinus = None
        self.tau1 = None
        self.tau2 = None

                                        
        self.prediction_history: List[Dict[str, Dict[str, float]]] = []
        self._current_prediction_sums: Optional[Dict[Tuple[str, str], float]] = None
        self._current_prediction_counts: Optional[Dict[Tuple[str, str], int]] = None
        self._global_prediction_totals: Dict[Tuple[str, str], float] = {}
        self._previous_predictions: Dict[str, Dict[str, float]] = {}
        self.current_prediction = 0.0
        self.cumulative_prediction = 0.0
        self.predictorSum = 0.0
        self.predictorCount = 0
        self._trial_predictions: Dict[str, Dict[str, float]] = {}

                                           
        self.predictors: Optional[np.ndarray] = None
        self.retainErrors = True
        self.csTotalErrors: Dict[str, float] = {}
        self.usTotalErrors: Dict[str, float] = {}
        self.csTotalSaliences: Dict[str, float] = {}
        self.usTotalSaliences: Dict[str, float] = {}
        self.csAverageErrors: Dict[str, float] = {}
        self.usAverageErrors: Dict[str, float] = {}
        self.trialAppearances: Dict[str, int] = {}
        self.trialTypes: Dict[str, int] = {}
        self.trialIndexMap: Dict[str, Dict[int, int]] = {}
        self.trialIndexMapb: Dict[str, Dict[int, int]] = {}
        self.trialTypeCounterMap: Dict[str, int] = {}
        self.trialAppearances2: Dict[str, int] = {}
        self.phases_appearances: Dict[str, int] = {}
        self.distractors: Dict[str, List[str]] = {}
        self.probeTrials: List[int] = []
        self.probeTrialTypes: Dict[str, int] = {}
        self.probeIndexes: Dict[str, int] = {}
        self.probeTiming: Dict[str, Dict[str, Tuple[int, int]]] = {}
        self.nameIndexes: Dict[str, int] = {}
        self.ptime: Dict[str, int] = {}
        self.ptime2: Dict[str, int] = {}
        self.objectList: List = []
        self.objectMap: Dict[Any, Any] = {}
        self.presentCS: Set[str] = set()
        self.presentStimuli: List = []
        self.csActiveThisTrial: Set[str] = set()
        self.probeCSActiveThisTrial: Set[str] = set()
        self.activeCS: List = []
        self.activeList: List = []
        self.activeLastStep: List = []
        self.csActiveLastStep: List = []
        self.results: Dict[str, Any] = {}
        self.cues: Dict[str, Any] = {}
        self.contextCfgs: Dict[str, Any] = {}
        self.contextCfgs[context.get_symbol()] = context
        self.usPredictions: Dict[str, float] = {}
        self.csPredictions: Dict[str, float] = {}
        self.averageCSErrors: Dict[str, float] = {}
        self.elementPredictions: Optional[np.ndarray] = None
        self.trialLengths: List[int] = []
        self.completeLengths: List[int] = []
        self.allSequences: List = []
        self.originalSequence: List = []
        self.tempMap: Dict[Any, Any] = {}
        self.allMap: Dict[Any, Any] = {}
        self.usIndexes: Dict[str, int] = {}
        self.csIndexes: Dict[str, int] = {}
        self.timePointElementErrors: Optional[np.ndarray] = None
        self.lastTrialElementErrors: Optional[np.ndarray] = None
        self.onsetMap: Dict[str, int] = {}
        self.offsetMap: Dict[str, int] = {}
        self.generalOnsetMap: Dict[str, int] = {}
        self.generalOffsetMap: Dict[str, int] = {}
        self.csMap: Dict[str, List] = {}
        self.currentSeq = ""
        self.nextSeq = ""
        self.currentTrial = ""
        self.iti = 0
        self.control = None
        self.maxMaxOnset = 0

                             
        self.cume = 0.0

                                                     
        if self.group is not None:
            for entry_key, entry_value in self.group.get_cues_map().items():
                if entry_key in seq:
                    self.cues[entry_key] = entry_value

    def _record_prediction_contribution(self, source: str, target: str, value: float):
        if self._current_prediction_sums is None or self._current_prediction_counts is None:
            return
        key = (source, target)
        self._current_prediction_sums[key] += value
        self._current_prediction_counts[key] += 1
        self._global_prediction_totals[key] = self._global_prediction_totals.get(key, 0.0) + value

    def _record_live_prediction(self, source: str, target: str, value: float) -> None:
        if source not in self._trial_predictions:
            self._trial_predictions[source] = {}
        prev = self._trial_predictions[source].get(target)
        if prev is None or abs(value) > abs(prev):
            self._trial_predictions[source][target] = value

    def _capture_prediction_details(self, target_stim: str, source_stim: str,
                                    target_element: 'StimulusElement',
                                    source_element: 'StimulusElement',
                                    value: float) -> None:
        if target_stim not in self._latest_element_contributions:
            self._latest_element_contributions[target_stim] = {}

        element_idx = target_element.getMicroIndex()
        if element_idx not in self._latest_element_contributions[target_stim]:
            self._latest_element_contributions[target_stim][element_idx] = {}

        self._latest_element_contributions[target_stim][element_idx][source_stim] = (
            self._latest_element_contributions[target_stim][element_idx].get(source_stim, 0.0)
            + value
        )

    def _finalize_trial_prediction_stats(self, trial_index: int):
        if self._current_prediction_sums is None or self._current_prediction_counts is None:
            return

        averages: Dict[str, Dict[str, float]] = defaultdict(dict)
        for (source, target), total in self._current_prediction_sums.items():
            count = max(1, self._current_prediction_counts[(source, target)])
            averages[source][target] = total / count

        self.prediction_history.append({f"trial_{trial_index}": dict(averages)})

                                                    
        self._current_prediction_sums = defaultdict(float)
        self._current_prediction_counts = defaultdict(int)

    def algorithm(self, sequence: List, temp_res: Dict, context: bool,
                  probe_results2: Dict):

        print(f"DEBUG: algorithm called with {len(sequence)} trials")
        
                                                             
        self.trialAppearances.clear()
        self.trialTypes.clear()
        
        for i in range(len(sequence)):
            self.trialTypes[sequence[i].to_string()] = 1
        
        for s in self.trialTypes.keys():
            for i in range(len(sequence)):
                if sequence[i].to_string() == s:
                    if s in self.trialAppearances:
                        self.trialAppearances[s] = self.trialAppearances[s] + 1
                    else:
                        self.trialAppearances[s] = 1
        
                                                    
        self.csActiveLastStep.clear()
        self.activeLastStep.clear()
        
                                                
        if context:
            temp_context = self.group.get_cues_map().get(self.contextCfg.get_symbol())
            if temp_context is not None:
                self.group.get_cues_map()[temp_context.get_name()] = temp_context
        
        self.activeList.clear()
        self.csActiveThisTrial.clear()
        self.probeCSActiveThisTrial.clear()
        
                                                                   
        most_elements = 0
        for stim in self.group.get_cues_map().values():
            most_elements = max(most_elements, len(stim.get_list()))
            stim.set_context_reset(self.contextReset)
            
            for se in stim.get_list():
                if stim.is_us or se.isUS:
                    se.setIntensity(self.intensity)
                se.setCSCLike(self.group.get_model().get_skew(False) if self.group.get_model() else 2.0)
                se.setVartheta(self.vartheta)
                se.setSubsetSize(self.setsize)
            
            if self.trials == 0:
                stim.set_zero_probe()
        
        for com in self.group.get_cues_map().values():
            if len(com.get_name()) > 1:
                a = self.group.get_cues_map().get(com.get_name()[1])
                b = self.group.get_cues_map().get(com.get_name()[2])
                com.initialize(a, b)
        
        for stimulus in self.group.get_cues_map().values():
            for element in stimulus.get_list():
                element.setPhase(0)
        
                                                             
        self.elementPredictions = np.zeros((len(self.group.get_cues_map()), most_elements), dtype=np.float32)

                                                                       
        last_element_preds = getattr(self.group, 'last_element_predictions', None)
        if last_element_preds:
            for idx, name in enumerate(self.group.get_cues_map().keys()):
                if name in last_element_preds:
                    prev = last_element_preds[name]
                    length = min(len(prev), self.elementPredictions.shape[1])
                    self.elementPredictions[idx, :length] = prev[:length]
                                                                      
        last_stim_preds = getattr(self.group, 'last_predictions', None)
        self._previous_predictions = {}
        if last_stim_preds:
            self._previous_predictions = {
                src: {tgt: float(val) for tgt, val in targets.items()}
                for src, targets in last_stim_preds.items()
            }

                                                                
        if self.group.get_model().is_errors:
            self.timePointElementErrors = np.zeros((self.trials * self.sessions,
                                                    len(self.group.get_cues_map()),
                                                    most_elements), dtype=np.float32)
        
        if self.group.get_model().is_errors2:
            first_stim = list(self.group.get_cues_map().values())[0]
            self.lastTrialElementErrors = np.zeros((first_stim.get_all_max_duration() *
                                                    len(self.group.get_cues_map()),
                                                    len(self.group.get_cues_map()),
                                                    most_elements), dtype=np.float32)
        
                                              
        for i in range(1, self.trials * self.sessions + 1):
            if self.control and self.control.is_cancelled():
                break
            
                                                              
            if i % self.trials == 1 and i != 1:
                self.itis.reset()
                for stimulus in self.group.get_cues_map().values():
                    if stimulus.is_context:
                        stimulus.reset_activation(True)
            
                                  
            self.csActiveThisTrial.clear()
            self.probeCSActiveThisTrial.clear()
            
            count = 0                                                       
            
                                                                     
            self.currentSeq = sequence[(i - 1) % self.trials].to_string()
            self.nextSeq = ("nextPhase" if i == self.trials * self.sessions 
                          else sequence[i % (self.trials if self.trials != 0 else 1)].to_string())
            
                                                                     
            for stim in self.group.get_cues_map().values():
                for elem in stim.get_list():
                    elem.setCurrentTrialString(self.currentSeq)
                    elem.setNextString(self.nextSeq)
            
                                                 
            if (i) // self.trials >= self.sessions - 1:
                last = True
                for k in range(1, self.trials * self.sessions - i + 1):
                    if sequence[(k + i - 1) % self.trials].to_string() == self.currentSeq:
                        last = False
                
                if last:
                    self.probeIndexes[self.currentSeq] = i
            
                                                                                   
            current_trial = sequence[(i - 1) % self.trials]
            cur_name_st = current_trial.to_string()
            if cur_name_st not in self.probeTiming:
                self.probeTiming[cur_name_st] = {}
            
                                                     
            self._current_prediction_sums = defaultdict(float)
            self._current_prediction_counts = defaultdict(int)
            self._trial_predictions = {
                src: targets.copy() for src, targets in self._previous_predictions.items()
            }
            self._latest_element_contributions = {}

                                                          
            self.tempMap.clear()
            self.allMap.clear()
            
            trial = sequence[(i - 1) % self.trials].copy()
            self.presentCS.update(trial.get_cues())
            
            for cs in trial.get_cues():
                self.tempMap[cs] = temp_res.get(cs.get_name())
            
            for j in range(len(sequence)):
                a_trial = sequence[j].copy()
                for cs in a_trial.get_cues():
                    if cs not in self.allMap:
                        self.allMap[cs] = temp_res.get(cs.get_name())
            
                                                      
            self.iti = int(round(self.itis.next() / 
                                self.group.get_controller().get_model().get_timestep_size()))
            
            timings = self.timingConfig.make_timings(set(self.tempMap.keys()))
            trial_length = timings.get_total()[1]
                                                                                     
                                                                                            
            self.group.trial_length = trial_length
            us_onset = timings.get_us()[0]
            us_offset = timings.get_us()[1]
            
            if (i % self.trials) <= self.trials:
                self.maxMaxOnset = max(self.maxMaxOnset, timings.get_cs_total()[1])
            
                                                      
            self.trialLengths.append(trial_length)
            self.completeLengths.append(trial_length + self.iti)
            
                                                            
            self.usIndexes.clear()
            self.csIndexes.clear()
            us_index_count = 0
            number_common = 0
            
            for s in self.group.get_cues_map().values():
                if (self.group.get_first_occurrence(s) >= 0 and 
                    self.group.get_first_occurrence(s) < self.get_phase_num()):
                    
                    if s.is_us:
                        self.usIndexes[s.get_name()] = us_index_count
                    elif not s.is_common():
                        self.csIndexes[s.get_name()] = us_index_count
                    elif s.is_common():
                        self.csIndexes[s.get_name()] = us_index_count
                        number_common += 1
                
                us_index_count += 1
                
                                                                               
                names = list(s.get_name())
                css = [None] * len(names)
                onset = -1
                offset = trial_length
                counter = 0
                
                for character in names:
                    if character == 'c':
                        pass
                    else:
                        for cs in self.tempMap.keys():
                            if cs.get_name() == character:
                                css[counter] = cs
                        
                                                                                         
                                                                                               
                        if css[counter]:
                            timing_result = timings.get(css[counter])
                            if timing_result and timing_result != [-1, -1]:
                                temp_onset = timing_result[0]
                                temp_offset = timing_result[1]
                            else:
                                temp_onset = -1
                                temp_offset = -1
                        else:
                            temp_onset = -1
                            temp_offset = -1
                        
                        onset = max(temp_onset, onset)
                        offset = min(temp_offset, offset)
                        counter += 1
                
                general_onset = -1
                general_offset = -1
                
                self.csMap[s.get_name()] = css
                self.onsetMap[s.get_name()] = onset
                self.offsetMap[s.get_name()] = offset
                self.generalOnsetMap[s.get_name()] = general_onset
                self.generalOffsetMap[s.get_name()] = general_offset
            
                                                                 
            for s in self.group.get_cues_map().values():
                if "c" in s.get_name():
                    name1 = s.get_name()[1]
                    name2 = s.get_name()[2]
                    onset = max(0, min(self.onsetMap[name1], self.onsetMap[name2]))
                    offset = max(self.offsetMap[name1], self.offsetMap[name2])
                    
                    self.onsetMap[s.get_name()] = onset
                    self.offsetMap[s.get_name()] = offset
            
                                                
            for j in range(1, trial_length + self.iti):
                if self.control and self.control.is_cancelled():
                    break
                
                self.activeList.clear()
                self.activeCS.clear()
                print(f"DBG  Phase {self.phaseNum}")
                                                                                   
                for stimulus in self.group.get_cues_map().values():
                    names = list(stimulus.get_name())
                    css = self.csMap[stimulus.get_name()]
                    onset = self.onsetMap[stimulus.get_name()]
                    offset = self.offsetMap[stimulus.get_name()]
                    general_onset = self.generalOnsetMap[stimulus.get_name()]
                    general_offset = self.generalOffsetMap[stimulus.get_name()]
                    cs_name = stimulus.get_name()
                    
                    stimulus.set_trial_length(trial_length + self.iti)
                    active = (j >= onset and j <= offset)
                    us_active = (timings.get_us()[0] < j and j <= timings.get_us()[1])
                    
                                                                              
                    if stimulus.is_us:
                        if stimulus.get_name() != "+":
                            stimulus.set_duration(timings.get_us()[1] - timings.get_us()[0],
                                                us_onset, us_offset, j - timings.get_us()[0],
                                                us_active and current_trial.is_reinforced(), j)
                        else:
                            stimulus.set_duration(us_offset - us_onset, us_onset, us_offset,
                                                j - timings.get_us()[0],
                                                us_active and current_trial.is_reinforced(), j)
                        general_onset = us_onset
                        general_offset = us_offset
                    
                    elif self.contextCfg.get_context().to_string() == cs_name:
                        stimulus.set_duration(trial_length + self.iti, 0, trial_length + self.iti - 1,
                                            j, True, j)
                        general_onset = 0
                        general_offset = trial_length + self.iti - 1
                    
                    elif not stimulus.is_context:
                        print(f"DEBUG: About to call set_duration on {stimulus.get_name()}")
                        print(f"DEBUG: stimulus type: {type(stimulus)}")
                        print(f"DEBUG: stimulus.set_duration type: {type(stimulus.set_duration)}")
                        print(f"DEBUG: Arguments: dur={offset - onset}, onset={onset}, offset={offset}, duration_point={j - onset}, active={active}, real_time={j}")
                        try:
                            stimulus.set_duration(offset - onset, onset, offset, j - onset, active, j)
                            print(f"DEBUG: set_duration completed for {stimulus.get_name()}")
                        except Exception as e:
                            print(f"ERROR: set_duration failed on {stimulus.get_name()}: {e}")
                            print(f"ERROR: stimulus type: {type(stimulus)}")
                            print(f"ERROR: stimulus.set_duration type: {type(stimulus.set_duration)}")
                            import traceback
                            traceback.print_exc()
                        general_onset = onset
                        general_offset = offset
                    
                    elif not (self.contextCfg.get_context().to_string() == cs_name) and stimulus.is_context:
                        stimulus.set_duration(trial_length + self.iti, 0, trial_length + self.iti - 1,
                                            j, False, j)
                        general_onset = 0
                        general_offset = trial_length + self.iti - 1
                    
                                                             
                    if stimulus.get_name() not in self.probeTiming[cur_name_st]:
                        self.probeTiming[cur_name_st][stimulus.get_name()] = (general_onset, general_offset)
                
                                                                            
                self.usPredictions.clear()
                self.csPredictions.clear()
                
                if self.elementPredictions is not None:
                                                         
                    for naming in self.usIndexes.keys():
                        current_us = self.group.get_cues_map()[naming]
                        temp_prediction = 0.0
                        div = len(current_us.get_list())
                        
                        for k2 in range(len(current_us.get_list())):
                            temp_prediction += abs((current_us.get(k2).getAsymptote() - 
                                                   self.elementPredictions[self.usIndexes[naming]][k2]) / div)
                        
                        if abs(temp_prediction) > 0.05:
                            self.usPredictions[naming] = temp_prediction
                    
                                                         
                    for naming in self.csIndexes.keys():
                        current_cs = self.group.get_cues_map()[naming]
                        temp_prediction = 0.0
                        div = len(current_cs.get_list())
                        
                        for k2 in range(len(current_cs.get_list())):
                            temp_prediction += abs((current_cs.get(k2).getAsymptote() - 
                                                   self.elementPredictions[self.csIndexes[naming]][k2]) / div)
                        
                        if abs(temp_prediction) > 0.05:
                            self.csPredictions[naming] = temp_prediction
                
                                                               
                average_error = 0.0
                for s in self.usPredictions.keys():
                    average_error += self.usPredictions[s] / float(len(self.usPredictions))
                
                                                                            
                for s in self.group.get_cues_map().keys():
                    temp_error = 0.0
                    correction = 0 if (self.group.get_cues_map()[s].is_context or 
                                      self.group.get_cues_map()[s].is_us) else number_common
                    factor = 0 if self.group.get_cues_map()[s].is_us else 1
                    
                    for s2 in self.csPredictions.keys():
                        if s != s2 and s not in s2 and s2 not in s:
                            temp_error += self.csPredictions[s2] / float(max(1, len(self.csPredictions) - factor - correction))
                    
                    self.averageCSErrors[s] = temp_error
                
                                                                   
                should_update_us = False
                for naming in self.usIndexes.keys():
                    if self.group.get_cues_map()[naming].get_should_update():
                        should_update_us = True
                
                should_update_cs = False
                for naming in self.csIndexes.keys():
                    if self.group.get_cues_map()[naming].get_should_update():
                        should_update_cs = True
                
                                                                             
                stim_count = 0
                for cue in self.group.get_cues_map().values():
                    cs_name = cue.get_name()
                    counter_bbb = 0
                    
                    for el in cue.get_list():
                        act = el.getGeneralActivation() * el.getParent().get_salience()
                        right_time = j % max(1, us_onset) <= (us_offset - us_onset)
                        right_time = False                       
                        
                                                                 
                        if cue.get_name() in ['A', 'B']:
                            print(f"    DEBUG {cue.get_name()} element {el.getMicroIndex()}: generalActivation={el.getGeneralActivation():.6f}, salience={el.getParent().get_salience():.6f}, act={act:.6f}")
                        
                                                                   
                        if should_update_us or True or len(self.usIndexes) == 0 or (cue.is_context and right_time):
                            el.storeAverageUSError(abs(average_error), act)
                        
                        if should_update_cs or True or len(self.csIndexes) == 0:
                            el.storeAverageCSError(abs(self.averageCSErrors[cue.get_name()]), act)
                        
                                                                  
                        old_prediction = self.elementPredictions[stim_count][el.getMicroIndex()] if self.elementPredictions is not None else 0.0
                        el.updateAssocTrace(max(0, old_prediction))
                        
                                                                    
                        if hasattr(self, 'initialSeq') and 'AB' in self.initialSeq and '+' in self.initialSeq:
                            if cue.get_name() == 'B':
                                print(f"    B element {el.getMicroIndex()}: using OLD prediction {old_prediction:.6f} for updateAssocTrace")
                        

                        sum_of_predictions = 0.0
                        prediction_breakdown = []
                        for cue2 in self.group.get_cues_map().values():
                            current_element_prediction = 0.0
                            for el2 in cue2.get_list():
                                                                                                  
                                if el2 == el or el2.getGeneralActivation() == 0.0:
                                    continue

                                if el2.getName() in el.names:
                                    prediction = el2.getPrediction(
                                        el2.names.index(el.getName()),
                                        el.getMicroIndex(),
                                        True, False
                                    )
                                current_element_prediction += prediction
                                self._record_prediction_contribution(cue2.get_name(), cue.get_name(), prediction)
                                self._record_live_prediction(cue2.get_name(), cue.get_name(), prediction)
                                self._capture_prediction_details(cue.get_name(), cue2.get_name(), el, el2, prediction)
                                if prediction != 0.0:
                                    prediction_breakdown.append(f"{cue2.get_name()}[{el2.getMicroIndex()}]: {prediction:.6f}")
                            sum_of_predictions += current_element_prediction
                        
                                                          
                        self.elementPredictions[stim_count][counter_bbb] = sum_of_predictions
                        
                                                                              
                        if hasattr(self, 'initialSeq') and 'AB' in self.initialSeq and '+' in self.initialSeq:
                            if cue.get_name() in ['A', 'B']:
                                print(f"    PREDICTION {cue.get_name()}[{counter_bbb}]: OLD={old_prediction:.6f}, NEW={sum_of_predictions:.6f}")
                                for breakdown in prediction_breakdown:
                                    print(f"      -> {breakdown}")
                                print(f"    PREDICTION {cue.get_name()}[{counter_bbb}] FINAL: {sum_of_predictions:.6f}")
                        else:
                                                        
                            if cue.get_name() in ['A', 'B']:
                                print(f"    PREDICTION {cue.get_name()}[{counter_bbb}]: OLD={old_prediction:.6f}, NEW={sum_of_predictions:.6f}")
                                for breakdown in prediction_breakdown:
                                    print(f"      -> {breakdown}")
                                print(f"    PREDICTION {cue.get_name()}[{counter_bbb}] FINAL: {sum_of_predictions:.6f}")
                        
                                                                               
                        if hasattr(self, 'initialSeq') and 'AB' in self.initialSeq and '+' in self.initialSeq:
                            if cue.get_name() == 'B':
                                print(f"    Storing prediction for B element {counter_bbb}: {sum_of_predictions:.6f}")
                                                                       
                                for cue2 in self.group.get_cues_map().values():
                                    contrib = 0.0
                                    for el2 in cue2.get_list():
                                                                                                          
                                        if el2 == el or el2.getGeneralActivation() == 0.0:
                                            continue
                                            
                                        if el2.getName() in el.names:
                                            prediction = el2.getPrediction(
                                                el2.names.index(el.getName()),
                                                el.getMicroIndex(),
                                                True, False
                                            )
                                            contrib += prediction
                                    if contrib > 0.001:
                                        print(f"      Contribution from {cue2.get_name()}: {contrib:.6f}")
                        
                        counter_bbb += 1
                    
                    self.csActiveThisTrial.add(cs_name)
                    self.activeList.append(cue)
                    stim_count += 1
                
                                                                       
                counter1 = 0
                for stim in self.group.get_cues_map().values():
                    if stim.get_name() not in self.nameIndexes:
                        self.nameIndexes[stim.get_name()] = counter1
                    
                    for el in stim.get_list():
                        if (j < timings.get_us()[1]) and self.group.get_model().is_errors:
                            self.timePointElementErrors[i - 1][counter1][el.getMicroIndex()] = (
                                (el.getAsymptote() - self.elementPredictions[counter1][el.getMicroIndex()]) / 
                                timings.get_us()[1]
                            )
                        
                        if i == self.trials * self.sessions and self.group.get_model().is_errors2:
                            self.lastTrialElementErrors[j - 1][counter1][el.getMicroIndex()] = (
                                el.getAsymptote() - self.elementPredictions[counter1][el.getMicroIndex()]
                            )
                        
                                                                               
                        average_error = el.getCurrentUSError()
                        average_cs_error = el.getCurrentCSError()
                        temp_dopamine = el.getVariableSalience()
                        temp_cs_dopamine = el.getCSVariableSalience()
                        
                        act = el.getGeneralActivation() * el.getParent().get_salience()
                        threshold = 0.3 if stim.is_context else 0.4
                        threshold2 = 0.9 if stim.is_context else 0.9
                        
                        right_time = j % max(1, us_onset) <= (us_offset - us_onset)
                        right_time = False
                        
                                                                       
                        if should_update_us or True or len(self.usIndexes) == 0 or (stim.is_context and right_time):
                            total_error_us = el.getTotalError(abs(average_error))
                            temp_dopamine = (temp_dopamine * (1 - self.integration * act) * 
                                           (1 - act * (total_error_us / 100.0 if total_error_us > threshold else 0)) + 
                                           (self.integration * act * max(0, min(1, abs(average_error))) * 
                                            max(el.getParent().get_was_active(), el.getGeneralActivation())))
                        
                        if should_update_cs or True or len(self.csIndexes) == 0:
                            total_error_cs = el.getTotalCSError(abs(average_cs_error))
                            temp_cs_dopamine = (temp_cs_dopamine * (1 - self.csIntegration * act) * 
                                              (1 - (total_error_cs / 100.0 if total_error_cs > threshold2 else 0)) + 
                                              self.csIntegration * act * max(0, min(1, abs(average_cs_error))) * 
                                              max(el.getParent().get_was_active(), el.getGeneralActivation()))
                        
                                                                     
                        if not stim.is_us:
                            el.setVariableSalience(temp_dopamine)
                        el.setCSVariableSalience(temp_cs_dopamine)
                    
                    counter1 += 1
                
                                                           
                print(f"DEBUG: About to call increment_timepoint for all stimuli at time {j}")
                for cl in self.group.get_cues_map().values():
                    print(f"DEBUG: Calling increment_timepoint on {cl.get_name()} at time {j}")
                    print(f"DEBUG: cl type: {type(cl)}, cl.get_name(): {cl.get_name()}")
                    print(f"DEBUG: cl.increment_timepoint type: {type(cl.increment_timepoint)}")
                    try:
                        cl.increment_timepoint(j, j > trial_length)
                        print(f"DEBUG: increment_timepoint completed for {cl.get_name()}")
                    except Exception as e:
                        print(f"ERROR: increment_timepoint failed on {cl.get_name()}: {e}")
                        print(f"ERROR: cl type: {type(cl)}")
                        print(f"ERROR: cl.increment_timepoint type: {type(cl.increment_timepoint)}")
                        import traceback
                        traceback.print_exc()
                
                                                                                         
                self.updateCues(0, temp_res, set(self.tempMap.keys()), j)
                
                                                               
                for cl in self.group.get_cues_map().values():
                    cl.reset_for_next_timepoint()
            
                                  
            self.activeLastStep.clear()
            self.csActiveLastStep.clear()

                                                            
            self._finalize_trial_prediction_stats(i)
            self._store_trial_predictions(i)

                                          
        self.group.compact_db()
        
        
                                             
        if 'i' in locals() and self.sessions > 0:
            if i % self.sessions == 0:
                if self.control:
                    self.control.increment_progress(1)
        
                           

                                                                           
        self._store_final_predictions()
        self._store_global_contributions()

    def _store_final_predictions(self):
        """Aggregate elementPredictions into stimulus-level values."""
        aggregated: Dict[str, Dict[str, float]] = {}

        if self.elementPredictions is None:
                                                            
            if getattr(self.group, 'prediction_history', None):
                self.group.last_predictions = {
                    src: targets.copy()
                    for src, targets in self.group.prediction_history[-1]['predictions'].items()
                }
            return
                                                                                       
        if getattr(self.group, 'prediction_history', None):
            self.group.last_predictions = {
                src: targets.copy()
                for src, targets in self.group.prediction_history[-1]['predictions'].items()
            }

    def _store_trial_predictions(self, trial_index: int):
        if self.elementPredictions is None:
            return
 
        stimuli = list(self.group.get_cues_map().values())
        trial_snapshot: Dict[str, Dict[str, float]] = {}
 
        for stim_index, stimulus in enumerate(stimuli):
            stim_name = stimulus.get_name()
            trial_snapshot.setdefault(stim_name, {})
            for target_name in stimulus.names:
                total_pred = 0.0
                target_idx = stimulus.names.index(target_name)
                for element in stimulus.get_list():
                    try:
                        total_pred += element.getPrediction(target_idx, element.getMicroIndex(), True, False)
                    except Exception:
                        continue
                trial_snapshot[stim_name][target_name] = total_pred
 
        trial_snapshot['__element_details__'] = {}
        trial_snapshot['__element_details__'] = self._latest_element_contributions

        if not hasattr(self.group, 'prediction_history'):
            self.group.prediction_history = []

        entry = {
            'trial': trial_index,
            'global_trial': len(self.group.prediction_history) + 1,
            'predictions': {src: tgt.copy() for src, tgt in self._trial_predictions.items()},
            'element_predictions': trial_snapshot.get('__element_details__', {}),
            'details': trial_snapshot.get('__element_details__', {})
        }
 
                                                                               
        norm_predictions = {
            src: {tgt: float(val) for tgt, val in targets.items()}
            for src, targets in entry['predictions'].items()
        }

        self.group.prediction_history.append(entry)
        self._previous_predictions = {
            source: targets.copy() for source, targets in norm_predictions.items()
        }
        self.group.last_element_predictions = {
            name: np.copy(self.elementPredictions[idx])
            for idx, name in enumerate(self.group.get_cues_map().keys())
        }
        last_preds_copy: Dict[str, Dict[str, float]] = {}
        for src, targets in norm_predictions.items():
            copy_targets = {tgt: float(val) for tgt, val in targets.items()}
            copy_targets.setdefault('+', float(targets.get('+', 0.0)))
            last_preds_copy[src] = copy_targets
        self.group.last_predictions = last_preds_copy
 
        if hasattr(self.group, 'model') and hasattr(self.group.model, 'prediction_history'):
            self.group.model.prediction_history.append(entry)

                                                                   
        self._trial_predictions = {}

    def _store_global_contributions(self):
        """Persist accumulated source->target contributions to the group."""
        aggregated: Dict[str, Dict[str, float]] = {}

        for (source, target), total in self._global_prediction_totals.items():
            aggregated.setdefault(source, {})[target] = total

    
    def updateCues(self, beta_error: float, temp_res: Dict, cs_set: Set, time: int):
        first_count = 0
        for cue in self.group.get_cues_map().values():
            count2 = 0
            for cue2 in self.group.get_cues_map().values():
                                          
                if cue2.get_name() not in cue.get_names():
                    cue.get_names().append(cue2.get_name())
                
                                                                      
                for element_idx, el in enumerate(cue.get_list()):
                    for target_idx, el2 in enumerate(cue2.get_list()):
                                                                                         
                        pred1 = self.elementPredictions[first_count][element_idx]
                        pred2 = self.elementPredictions[count2][target_idx]
                        error1 = (el.getAsymptote() - pred1)
                        error2 = (el2.getAsymptote() - pred2)
                        
                                                                            
                        if hasattr(self, 'initialSeq') and 'AB' in self.initialSeq and '+' in self.initialSeq:
                            if cue.get_name() == 'B' and cue2.get_name() == '+':
                                print(f"    DEBUG B->+ weight update:")
                                print(f"      B element {el.getMicroIndex()}: asymptote={el.getAsymptote():.3f}, prediction={pred1:.3f}, error1={error1:.3f}")
                                print(f"      US element {el2.getMicroIndex()}: asymptote={el2.getAsymptote():.3f}, prediction={pred2:.3f}, error2={error2:.3f}")
                        
                                                                         
                        
                                                       
                        el.updateElement(el2.getDirectActivation(),
                                       el2.getAlpha(),
                                       el2,
                                       error1,
                                       error2,
                                       cue2.get_name(),
                                       self.group)
                count2 += 1
            first_count += 1
    
    
    def run(self):
        """Alias for runSimulator for compatibility."""
        return self.runSimulator()
    
    def runSimulator(self):
        self.results = dict(self.cues)
        context = self.group.get_model().is_use_context()
        combinations = self.group.get_model().get_combination_no() if self.random else 1
        
        if not hasattr(self.group, '_shared_rng'):
            self.group._shared_rng = random.Random(647926)                                       
        generator = self.group._shared_rng
        gn = self.group.get_name_of_group()
        
        self.group.make_map(f"{gn}{self.phaseNum} r")
        self.group.make_map(f"{gn}{self.phaseNum} r2")
        self.group.make_map(f"{gn}{self.phaseNum} rA")
        self.group.make_map(f"{gn}{self.phaseNum} rA2")
        self.group.make_map(f"{gn}{self.phaseNum} cur")
        self.group.make_map(f"{gn}{self.phaseNum} cur2")
        self.group.make_map(f"{gn}{self.phaseNum} curA")
        self.group.make_map(f"{gn}{self.phaseNum} curA2")

        self.originalSequence = []
        count = 0
        for t in self.orderedSeq:
            trial_tmp = t.copy()
            trial_tmp.set_trial_number(count)
            self.originalSequence.append(trial_tmp)
            count += 1
        
        self._build_trial_type_maps()
        
        for s in self.group.get_cues_map().values():
            s.set_reset_context(self.resetContext)
            for se in s.get_list():
                se.setCSCV(self.std)
                se.setUSCV(self.group.get_model().get_us_cv())
                se.setUSScalar(self.usScalar)
                se.setCSScalar(self.csScalar)
                se.setUSPersistence(self.usPersistence)
        
                                               
        print(f"DEBUG: Starting {int(combinations)} combinations with seed {self.group._shared_rng.getstate()[1][0] if hasattr(self.group, '_shared_rng') else 'unknown'}")
        for i in range(int(combinations)):
            if self.control and self.control.is_cancelled():
                break
            
            temp_seq = list(self.orderedSeq)
            
                                                    
            if self.random:
                for x in range(self.trials):
                    if len(self.orderedSeq) > 1:
                        nr = generator.randint(0, len(self.orderedSeq) - 2)
                        swap = temp_seq[x]
                        temp_seq.pop(x)
                        temp_seq.insert(nr, swap)
                
                                                      
                if i == 0:
                    for ran in range(10):
                        for x in range(self.trials):
                            if len(self.orderedSeq) > 1:
                                nr = generator.randint(0, len(self.orderedSeq) - 2)
                                swap = temp_seq[x]
                                temp_seq.pop(x)
                                temp_seq.insert(nr, swap)
            
                                                                    
            for s in self.group.get_cues_map().values():
                s.set_phase(self.phaseNum - 1)
                if i > 0:
                    s.increment_combination()
                if self.random:
                    s.reset(i + 1 == combinations, 0 if i == 0 else self.trials * self.sessions)
            
                                           
            temp_res = dict(self.cues)
            temp_probe_res = {}
            self.algorithm(temp_seq, temp_res, context, temp_probe_res)
            self.allSequences.append(temp_seq)
            
                                        
            self.group.push_cache()
            
                                                                           
                                                             
            self._process_combination_results(i, int(combinations))
            
                                                   
            try:
                self.timingConfig.advance()
            except Exception:
                pass
            self.timingConfig.restart_onsets()
            self.itis.reset()
        
                                               
        self._finalize_results()
        
                                      
        if not (self.control and self.control.is_cancelled()):
            self.cues.update(self.results)
    
    def _build_trial_type_maps(self):
        """Build trial type index maps."""
        canonical_names = {}
        trial_type_counters = {}
        self.trialTypeCounterMap.clear()
        self.trialIndexMap.clear()
        self.trialIndexMapb.clear()
        
        trial_type_array = []
        for i in range(len(self.initialSeq.split("/"))):
            s_num = self.initialSeq.split("/")[i]
            s_num = ''.join(filter(str.isdigit, s_num))
            s_type = ''.join(filter(str.isalpha, s_num))
            
            index = 1 if not s_num else float(s_num)
            trial_type = self.initialSeq.split("/")[i].replace(s_num, "")
            for j in range(int(index)):
                trial_type_array.append(trial_type)
            
                                                                                       
            trial_type_counters[trial_type] = 0
            self.trialTypeCounterMap[trial_type] = 0
            self.trialIndexMap[trial_type] = {}
            self.trialIndexMapb[trial_type] = {}
        
        for i in range(len(self.orderedSeq)):
            counter = trial_type_counters[trial_type_array[i]]
            trial_type_counters[trial_type_array[i]] = counter + 1
            self.trialTypeCounterMap[trial_type_array[i]] = counter + 1
            self.trialIndexMap[trial_type_array[i]][counter] = i
            self.trialIndexMapb[trial_type_array[i]][i] = counter
    
    def _process_combination_results(self, combination_num: int, total_combinations: int):
        """Process results after each combination"""
                                                                        
        pass
    
    def _finalize_results(self):
        """Finalize all results after all combinations"""
                                                                       
        pass
    
                         
    def get_phase_num(self) -> int:
        return self.phaseNum
    
    def get_no_trials(self) -> int:
        return self.trials
    
    def get_ordered_seq(self) -> List:
        return self.orderedSeq
    
    def get_results(self) -> Dict:
        return self.results
    
    def get_iti(self):
        return self.itis
    
    def is_random(self) -> bool:
        return self.random
    
    def set_control(self, control):
        self.control = control
    
    def set_beta_plus(self, beta: float):
        self.betaPlus = beta
    
    def set_lambda_plus(self, lambda_val: float):
        self.lambdaPlus = lambda_val
    
    def set_lambda_minus(self, lambda_val: float):
        self.lambdaMinus = lambda_val
    
    def set_vartheta(self, v: float):
        self.vartheta = v
        
    def initialize_stimuli(self):
        """Initialize stimuli for this phase."""
                                                        
                                                                       
        pass
