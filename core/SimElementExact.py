import numpy as np
from typing import List, Dict, Any, Optional
import random


class StimulusElement:
    def __init__(self, micro_index: int, parent, group, name: str,
                 alpha: float, std: float, trials: int, total_stimuli: int,
                 total_micros: int, total_max: int, generalization: float,
                 lambda_plus: float, us_boost: float, vartheta: float,
                 presence_mean: bool):
                          
        self.name = name
        
                                                                      
                                                            
        if not hasattr(group, '_shared_rng'):
            raise RuntimeError("Shared RNG must be set on group before creating StimulusElement")
        self.rng = group._shared_rng
        self.timepoint = 0
        self.alphaR = alpha
        self.std = std if std is not None else 1.0                            
        self.microstimulusIndex = micro_index
        self.activation = 0.0
        self.directActivation = 0.0
        self.phase = 0
        self.totalStimuli = total_stimuli
        self.trialCount = 0
        self.totalMax = total_max
        self.vartheta = vartheta
        if self.vartheta is None:
            print(f"WARNING: vartheta is None for {self.name}")
            self.vartheta = 0.5                 
        self.presenceMean = presence_mean
        
                                                                  
        self.random_midpoint = None                                       
        self.random_std = 0.4                                          
        
                           
        self.parent = parent
        self.group = group
        self.totalTrials = trials + 1
        
                                                                 
        self.trialTypeCount = {}
        self.session = 0
        self.combination = 1.0
        self.subelementNumber = 10.0
        self.totalElements = 1.0
        self.ratio = 1.0
        self.ctxratio = 1.0
        self.microPlus = 0.0
        self.microIndex = 0.0
        self.denominator = 1.0
        self.assoc = 0.0
        self.generalActivation = 0.0
        self.maxActivation = 0.0
        self.newValue = 0.0
        self.durationPoint = 0.0
        self.time = 0.0
        self.difference = 0.0
        self.numerator = 0.0
        self.intensity = 1.0
        
                                                        
        self.variableSalience = np.zeros(10, dtype=np.float32)                                         
        self.csVariableSalience = np.zeros(10, dtype=np.float32)
        self.totalUSError = np.zeros(10, dtype=np.float32)
        self.totalCSError = np.zeros(10, dtype=np.float32)
        self.presences = np.zeros(total_stimuli, dtype=np.float32)
        
                                  
        self.elementCurrentWeightsKey = f"{group.get_name_of_group()}{name}{micro_index}elementCurrentWeights"
        self.eligibilitiesKey = f"{group.get_name_of_group()}{name}{micro_index}eligibilities"
        self.aggregateActivationsKey = f"{group.get_name_of_group()}{name}{micro_index}aggregateActivations"
        self.aggregateSaliencesKey = f"{group.get_name_of_group()}{name}{micro_index}aggregateSaliences"
        self.aggregateCSSaliencesKey = f"{group.get_name_of_group()}{name}{micro_index}aggregateCSSaliences"
        
                                                                     
        self.subelementWeights = None
        self.oldElementWeights = None
        
                     
        self.firstPass = True
        self.isStored = False
        self.wasActive = False
        self.hasReset = False
        self.subsetSet = False
        self.disabled = False
        self.kickedIn = False
        self.isA = False
        self.isB = False
        self.notTotal = True
        self.outOfBounds = False
        
                             
        self.alphaN = alpha
        self.beta = 0.1                       
        self.salience = 0.5                       
        self.intensity = 1.0
        self.usSalience = 0.1
        self.startingAlpha = 1.0
        self.cscLikeness = 20.0
        self.USCV = 2.5
        self.usPersistence = 0.0
        
                                              
        self.names: List[str] = []
        
                                                               
        self.subelement_weights: Optional[np.ndarray] = None
        self.old_element_weights: Optional[np.ndarray] = None
        
                                
        self.isUS = self._is_us_name(name)
        self.isContext = self._is_context_name(name)
        
                                          
        self.adj = 8 if self.isUS else 0
        
        if self.isUS:
            self.names.append(name)
        
                                                                        
                                                                   
        if not self.isContext:                                               
            self.names.append("Context")
        
                                                                             
                                                                           
                                                           
                                                                 
    
    def set_all_stimulus_names(self, names: List[str]):
        """Set the names of all stimuli that this element can predict."""
        self.names = names.copy()
    
    def initialize_weights(self, subelement_number: int, total_stimuli: int, total_max: int):
        if self.subelement_weights is None:
            self.subelement_weights = np.zeros((int(subelement_number), total_stimuli, total_max), dtype=np.float32)
            self.old_element_weights = np.zeros((int(subelement_number), total_stimuli, total_max), dtype=np.float32)
        
                                       
        self.assoc = 0.0
        self.generalActivation = 0.0
        self.averageUSError = 0.0
        self.oldUSError = 0.0
        self.averageCSError = 0.0
        self.oldCSError = 0.0
        self.asymptote = 0.0
        
                
        self.presences = np.zeros(total_stimuli, dtype=np.float32)
        self.totalUSError = np.zeros(self.group.get_no_of_phases(), dtype=np.float32)
        self.totalCSError = np.zeros(self.group.get_no_of_phases(), dtype=np.float32)
        self.variableSalience = np.zeros(self.group.get_no_of_phases(), dtype=np.float32)
        self.csVariableSalience = np.zeros(self.group.get_no_of_phases(), dtype=np.float32)
        
                                                    
        self.subelementWeights: Optional[np.ndarray] = None
        self.oldElementWeights: Optional[np.ndarray] = None
        self.subelementActivations: Optional[np.ndarray] = None
        self.subelementNumber = 10.0
        self.totalElements = 1.0
        
                        
        self.trialTypeCount: Dict[str, int] = {}
        self.currentTrialString = ""
        self.nextTrialString = ""
        self.session = 0
        self.combination = 1.0
        
                             
        self.factor = 10.0
        self.adj = 8 if self.isUS else 0
        self.ratio = 1.0
        self.ctxratio = 1.0
        self.microPlus = 0.0
        self.microIndex = 0.0
        self.denominator = 1.0                                        
        self.durationPoint = 0.0
        self.maxActivation = 0.0
        
                     
        self.discount = 1.0
        self.exponent = 1.0
        self.dis = 0.0
        
                                      
        self.a: Optional[Any] = None
        self.b: Optional[Any] = None
        
                       
        self.elementCurrentWeightsKey = f"{self.group.get_name_of_group()}{self.parent.get_name()}{self.microstimulusIndex}elementCurrentWeights"
        self.eligibilitiesKey = f"{self.group.get_name_of_group()}{self.parent.get_name()}{self.microstimulusIndex}eligibilities"
        self.aggregateActivationsKey = f"{self.group.get_name_of_group()}{self.parent.get_name()}{self.microstimulusIndex}aggregateActivations"
        self.aggregateSaliencesKey = f"{self.group.get_name_of_group()}{self.parent.get_name()}{self.microstimulusIndex}aggregateSaliences"
        self.aggregateCSSaliencesKey = f"{self.group.get_name_of_group()}{self.parent.get_name()}{self.microstimulusIndex}aggregateCSSaliences"
        
                                  
        trials = self.group.get_total_trials() if hasattr(self.group, 'get_total_trials') else 10
        self._initialize_database_maps(self.group, trials, total_stimuli, total_max)

                                                                         
        no_of_phases = self.group.get_no_of_phases() if hasattr(self.group, 'get_no_of_phases') else 1

                                                                                                
        self.subelementActivations = np.zeros(int(subelement_number), dtype=np.float32)
        self.oldElementWeights = np.zeros((int(subelement_number), total_stimuli, total_max), dtype=np.float32)

                                           
        self.eligibilities = np.zeros(total_stimuli, dtype=np.float32)
        self.aggregateActivations = np.zeros(total_stimuli, dtype=np.float32)
        self.aggregateSaliences = np.zeros(total_stimuli, dtype=np.float32)
        self.aggregateCSSaliences = np.zeros(total_stimuli, dtype=np.float32)

                                          
        self.totalUSError = np.zeros(no_of_phases, dtype=np.float32)
        self.totalCSError = np.zeros(no_of_phases, dtype=np.float32)
        self.variableSalience = np.zeros(no_of_phases, dtype=np.float32)
        self.csVariableSalience = np.zeros(no_of_phases, dtype=np.float32)
        
                                                                      
        self.variableSalience[0] = self.alphaR
        self.csVariableSalience[0] = self.alphaN

    def _is_us_name(self, name: str) -> bool:
        """Check if name indicates US stimulus"""
        return "+" in name or name.upper() == "US"
    
    def _is_context_name(self, name: str) -> bool:
        """Check if name indicates context stimulus"""
        return name.upper() in ["CTX", "CONTEXT"] or "context" in name.lower()
    
    def _initialize_database_maps(self, group, trials: int, total_stimuli: int, total_max: int):
        iti = 0
        for sp in group.get_phases():
            iti = max(iti, int(sp.get_iti().get_minimum()))
        
        temp = np.zeros(total_max + iti, dtype=np.float32)
        
                              
        group.make_map(self.elementCurrentWeightsKey)
        group.make_map(self.eligibilitiesKey)
        group.make_map(self.aggregateActivationsKey)
        group.make_map(self.aggregateSaliencesKey)
        group.make_map(self.aggregateCSSaliencesKey)
        
                            
        for i in range(group.get_no_of_phases()):
            group.add_to_map(str(i), np.zeros((group.get_no_of_phases(), total_stimuli, total_max), dtype=np.float32),
                           self.elementCurrentWeightsKey, True)
        
        for i in range(total_stimuli):
            group.add_to_map(str(i), np.zeros(total_max, dtype=np.float32),
                           self.eligibilitiesKey, True)
        
        for i in range(trials + 1):
            group.add_to_map(str(i), np.zeros(total_max + iti, dtype=np.float32),
                           self.aggregateActivationsKey, True)
            group.add_to_map(str(i), np.zeros(total_max + iti, dtype=np.float32),
                           self.aggregateSaliencesKey, True)
            group.add_to_map(str(i), np.zeros(total_max + iti, dtype=np.float32),
                           self.aggregateCSSaliencesKey, True)
    
    def setSubsetSize(self, size: int):
        if self.subsetSet:
            return
        self.totalElements = float(size)
        
                                             
        self.discount = self.group.get_model().get_discount()
        self.dis = self.discount ** 10
        self.exponent = 0.0 if self.dis == 0 else abs(1.0 / self.dis - 1.0)

                                                                                          
        if hasattr(self.group.get_model(), 'get_discount'):
            self.discount = self.group.get_model().get_discount()
            self.dis = self.discount ** 10
            self.exponent = 0.0 if self.dis == 0 else abs(1.0 / self.dis - 1.0)
        
        self.outOfBounds = False
        if np.isnan(self.exponent) or self.exponent > 20:
            self.outOfBounds = True
        
        self.subsetSet = True
        
                                                              
        isCommon = len(self.getName()) > 1
        hasCommon = len(self.parent.get_common_map()) > 0
        
        commonProp = 0.0
        uniqueProp = 0.0
        
        if isCommon:
                                                                                
            try:
                if self.group.get_model() and hasattr(self.group.get_model(), 'get_common'):
                    common_dict = self.group.get_model().get_common()
                    if common_dict and self.group.get_name_of_group() in common_dict:
                        group_dict = common_dict.get(self.group.get_name_of_group())
                        if group_dict and self.getName() in group_dict:
                            commonProp = size * group_dict.get(self.getName())
                        else:
                            commonProp = size * 0.5
                    else:
                        commonProp = size * 0.5
                else:
                    commonProp = size * 0.5
            except:
                commonProp = size * 0.5                      
        
        if hasCommon:
                                                                                
            try:
                if self.group.get_model() and hasattr(self.group.get_model(), 'get_proportions'):
                    prop_dict = self.group.get_model().get_proportions()
                    if prop_dict and self.group.get_name_of_group() in prop_dict:
                        group_dict = prop_dict.get(self.group.get_name_of_group())
                        if group_dict and self.getName() in group_dict:
                            uniqueProp = size * (1.0 - group_dict.get(self.getName()))
                        else:
                            uniqueProp = size * 0.5
                    else:
                        uniqueProp = size * 0.5
                else:
                    uniqueProp = size * 0.5
            except:
                uniqueProp = size * 0.5                      
        
        commonValue = commonProp if isCommon else (uniqueProp if hasCommon else size)
        self.subelementNumber = float(np.floor(commonValue))
        
        if commonValue > 0:
            self.subelementNumber = max(1.0, self.subelementNumber)
        
                                                         
        if self.isContext:
            self.subelementNumber = 1.0
        
                                                                        
        if not self.isContext and self.subelementNumber < 10.0:
            self.subelementNumber = 10.0

                         
        se_count = int(self.subelementNumber)
        self.subelementWeights = np.zeros((se_count, self.totalStimuli, self.totalMax), dtype=np.float32)
        self.oldElementWeights = np.zeros((se_count, self.totalStimuli, self.totalMax), dtype=np.float32)
        self.subelementActivations = np.zeros(se_count, dtype=np.int32)
    
    def updateAssocTrace(self, assoc: float):
        if self.vartheta is None:
            print(f"ERROR: vartheta is None for {self.name} in updateAssocTrace")
            print(f"  self.name={self.name}, self.parent={self.parent.get_name() if self.parent else 'None'}")
            print(f"  assoc={assoc}, type(assoc)={type(assoc)}")
            self.vartheta = 0.5                 
        asoc = max(0.0, min(1.0, assoc * self.vartheta))
        self.assoc = asoc
        
                                                            
        count = 0
                                                                                           
        if asoc == 0.0 and hasattr(self, 'previous_assoc') and self.previous_assoc > 0:
            totalActivation = max(self.previous_assoc, self.directActivation)
        elif asoc == 0.0 and self.directActivation == 0.0:
            totalActivation = 0.01  
        else:
            totalActivation = max(asoc, self.directActivation)
        
                                                
        self.previous_assoc = asoc
        
                                                                
        while count < self.subelementNumber:
            i = self.randomWithRange(0, int(self.subelementNumber) - 1)
            
                                                                        
            if self.subelementActivations[i] == 0:
                count += 1
            
                                                                                                                                 
            self.subelementActivations[i] = 1 if (self.subelementActivations[i] == 1 or 
                                                  self.rng.random() < totalActivation) else 0
        
                                            
        self.generalActivation = max(0.0, max(self.assoc, self.activation, self.directActivation))
        
                                    
        temp_length = self.totalMax + self._get_iti()
        temp = np.zeros(temp_length, dtype=np.float32)
        ob3 = self.group.get_from_db(str(self.trialCount), self.aggregateActivationsKey)
        if ob3 is not None:
            temp = ob3
        if self.timepoint >= len(temp):
            new_temp = np.zeros(self.timepoint + 1, dtype=np.float32)
            new_temp[:len(temp)] = temp
            temp = new_temp
        temp[self.timepoint] = self.activation
        self.group.add_to_map(str(self.trialCount), temp, self.aggregateActivationsKey, True)
    
    def updateActivation(self, name: str, presence: float, duration: float, microstimulusIndex: int):
        self.microstimulusIndex = microstimulusIndex
        self.microPlus = microstimulusIndex + 1
                                          
        time = (1.0 - presence) if self.presenceMean else self.durationPoint
        
                                   
        if self.isUS:
            if time >= self.microIndex:
                time = max(0.0, time - self.usPersistence)
        
                                                            
        force_us_activation = False
        if self.isUS and presence > 0:
            force_us_activation = True
        
                                       
        difference = time - (self.ratio * (self.microIndex - 1.0))                             
        if difference < 0:
            difference *= self.cscLikeness                           
        if self.isUS and difference < 0:
            difference = 0.0                        
        numerator = difference ** 2                                                                                                      
        if self.isUS:               
            sqrt_term = np.sqrt((self.microPlus + self.adj) * self.USCV * self.ctxratio)
        else:                                             
            sqrt_term = np.sqrt((self.microPlus + self.adj) * self.std * self.ctxratio)

        denominator = 2.0 * (sqrt_term ** 2)
        
        if denominator > 0:
            newValue = np.exp(-numerator / denominator) if presence > 0 else 0.0
        else:
            newValue = 0.0

        if force_us_activation:
            newValue = 1.0
        
                                                                          
        if self.isContext:
            newValue = 0.3 if presence > 0 else 0.0
        
                                   
        self.maxActivation = max(self.maxActivation, newValue)
        
                                        
        self.activation = newValue * self.intensity
        
                                        
        self.activation = 0.0 if self.disabled else self.activation
        
                                           
        if self.getName() == name:
            if presence > 0:
                if force_us_activation:
                    self.directActivation = 1.0
                else:
                    self.directActivation = self.activation
            else:
                self.directActivation = 0.0
        
    
    def updateElement(self, otherActivation: float, otherAlpha: float,
                     other: 'StimulusElement', ownError: float, otherError: float,
                     otherName: str, group):
        """Debug version to track weight update sequence"""                       
        if self.generalActivation == 0.0:
            return
                            
        nE = otherError
        nE2 = abs(ownError)
        
                                                                  
        c1 = otherName in self.names
        
                                                                
        if c1:
            index_str = str(self.names.index(otherName))
        else:
            index_str = "0"

                                                    
        ob1 = group.get_from_db(index_str, self.eligibilitiesKey)
        temp = np.zeros(self.totalMax, dtype=np.float32)
        if ob1 is not None:
            temp = ob1
        
                       
        val1 = other.getAssoc() / (temp[other.getMicroIndex()] + 0.001) if c1 else 1.0
        
                       
        c2 = (1.0 if other.getAssoc() == 0 else val1) > 0.9
        
                       
        if c1 and c2:
            self.presences[self.names.index(otherName)] = other.parent.get_was_active()
        
                                                       
        if c1:
            idx = other.getMicroIndex()
            temp[idx] = max(temp[idx] * 0.95, other.getAssoc())
            group.add_to_map(index_str, temp, self.eligibilitiesKey, True)
        
                                                           
        if not self.outOfBounds:
            idx = other.getMicroIndex()
            base_eligi = 1.0 if other.getAssoc() == 0 else (other.getAssoc() / (temp[idx] + 0.001))
            eligi = base_eligi ** self.exponent
        else:
            eligi = 0.01
        
        if 0 < eligi < 0.01:
            eligi = 0.01
        
                                             
        asymptote1 = self.getAsymptote()
        asymptote2 = other.getAsymptote()
        ac1 = int(asymptote1)
        ac2 = int(asymptote2)
        
                                                                
        if (self.parent.get_name() == 'A' and other.isUS):
            print(f"DEBUG A->US asymptotes: self.asymptote={asymptote1:.6f} (int={ac1}), other.asymptote={asymptote2:.6f} (int={ac2})")
            print(f"DEBUG A->US errors: Own: {ownError:.6f}, other: {otherError:.6f})")
        
                                                
        selfDereferencer = 1.0
        if self.getName() in other.getName() or other.getName() in self.getName():
            selfDereferencer = 0.0 if self.isUS else 0.01
        
                                                     
        if otherName not in self.names:
            self.names.append(otherName)
        index = self.names.index(otherName)
        
                                                
        maxDurationPoint = self.durationPoint
        maxDurationPoint2 = other.getDurationPoint()
        
        fixer = 1.0
        if self.directActivation > 0.1:
            if maxDurationPoint >= maxDurationPoint2:
                fixer = 1.0
            else:
                fixer = self.parent.get_b()
        

        x1 = min(1.0, max(ac1, self.assoc * 0.9))
        x2 = min(1.0, max(ac2, other.getAssoc() * 0.9))
        x3 = (fixer * x2 - abs(x1 - x2)) / max(x1 + 0.001, x2 + 0.001)

                                                           
        nE = (otherError - ac2 * 1.0) + x3 * 1.0
        
                                                                     
        if (self.parent.get_name() == 'A' and other.isUS):
            print(f"DEBUG A->US calculation: x1={x1:.6f}, x2={x2:.6f}, x3={x3:.6f}, nE={nE:.6f}")
            print(f"DEBUG A->US errors: otherError={otherError:.6f}, ownError={ownError:.6f}")

                                                                  
        if other.isUS:
            self.asymptote = nE

                                        
        nE2 = ownError
        
                                                         
                                           
        currentVariableSalience = self.variableSalience[self.phase] if other.isUS else self.csVariableSalience[self.phase]
        
                                          
        curSal = self.beta if self.isUS else self.salience
        
                                         
        if self.isCommon():
            commonDiscount = self.subelementNumber / self.totalElements
        elif len(self.parent.get_common_map()) > 0:
            commonDiscount = 1.0 - self.group.get_model().get_proportions().get(
                self.group.get_name_of_group()).get(self.getName())
        else:
            commonDiscount = 1.0
        
                                                
                                                                                    
        other_gen_act = other.getGeneralActivation()
        other_dir_act = other.getDirectActivation()
        other_alpha = other.getAlpha()
        self_alpha = self.getAlpha()
        tempDelta = ((1.0 / (self.subelementNumber * len(self.parent.get_list()))) *
                     commonDiscount * self.generalActivation * self_alpha *
                     curSal * other_gen_act * eligi *
                     selfDereferencer * currentVariableSalience * nE)

        if self.isUS:
            tempDelta *= self.usSalience * self.intensity
        
                                      
        ownActivation = self.generalActivation
        decay = (1.0 - (np.sqrt(curSal) / 10000.0)) if ownActivation > 0.01 else 1.0
        
                                             
        totalWeight = 0.0
        totalPrediction = 0.0

        
        for i in range(int(self.subelementNumber)):
            oldWeight = self.subelementWeights[i][index][other.getMicroIndex()]
                                                                               
            newWeight = oldWeight * decay + tempDelta * self.subelementActivations[i]
                                                                                                                     
            bound = 2.0 / self.subelementNumber
            clippedWeight = max(-bound, min(bound, newWeight))
            self.subelementWeights[i][index][other.getMicroIndex()] = clippedWeight
            
                                                    
            totalWeight += self.subelementWeights[i][index][other.getMicroIndex()]
            totalPrediction += self.subelementWeights[i][index][other.getMicroIndex()] * self.generalActivation
                                                                 
        if group.get_model().is_external_save:
            storeLong = group.create_db_string(self, self.currentTrialString, other.getParent(),
                                              self.phase, self.session,
                                              self.trialTypeCount.get(self.currentTrialString, 0),
                                              self.timepoint, True)
            group.add_to_db(storeLong, totalPrediction)
            
            storeLong = group.create_db_string(self, self.currentTrialString, other.getParent(),
                                              self.phase, self.session,
                                              self.trialTypeCount.get(self.currentTrialString, 0),
                                              self.timepoint, False)
            group.add_to_db(storeLong, totalWeight)
        
                                                     
        ob2 = group.get_from_db(str(self.phase), self.elementCurrentWeightsKey)
        current = np.zeros((self.totalStimuli, self.totalMax), dtype=np.float32)
        if ob2 is not None:
            current = ob2
                                       
        if index < current.shape[0] and other.getMicroIndex() < current.shape[1]:
            current[index][other.getMicroIndex()] = totalWeight
        group.add_to_map(str(self.phase), current, self.elementCurrentWeightsKey, True)
        
                                                                
        if self.firstPass:
            self.timepoint += 1
            self.firstPass = False
    
    def getAsymptote(self) -> float:
        if self.timepoint <= self.parent.get_last_onset():
            if self.assoc > 0.9 or self.getDirectActivation() > 0.1 * self.intensity:
                asy = 1.0
            else:
                asy = 0.0
        else:
            if self.getDirectActivation() > 0.1 * self.intensity:
                asy = 1.0
            else:
                asy = 0.0
        return asy
    
    def get_dynamic_asymptote(self) -> float:
        return self.getAsymptote()
    
    def getPrediction(self, stimulus: int, element: int, current: bool, maximum: bool) -> float:
        totalPrediction = 0.0
        
        for i in range(int(self.subelementNumber)):
                                        
                                                                                                                                     
            activationFactor = 1.0 if self.directActivation > 0.1 else self.generalActivation * self.vartheta
            totalPrediction += self.subelementWeights[i][stimulus][element] * activationFactor
        
                                                        
                                                                       
        if hasattr(self.group, 'get_model') and self.group.get_model():
            if hasattr(self.group.get_model(), 'restrict_predictions') and self.group.get_model().restrict_predictions:
                totalPrediction = max(0.0, totalPrediction)                    
            else:
                                                                       
                                                                            
                pass
        
        return totalPrediction
    
    def setTrialLength(self, trialLength: int):                                                        
        self.microPlus = self.microstimulusIndex + 1.0
        self.microIndex = self.microPlus                             
        if self.ctxratio is None:
            self.ctxratio = 1.0
        if self.USCV is None:
            self.USCV = 2.5
        if self.std is None:
            self.std = 1.0
        if self.microPlus is None:
            self.microPlus = 0.0
        if self.adj is None:
            self.adj = 8 if self.isUS else 0
        
                                                                                                                              
        if self.isUS:
                                                           
            sqrt_term = np.sqrt((self.microPlus + self.adj) * self.USCV * self.ctxratio)
        else:
                                                              
            sqrt_term = np.sqrt((self.microPlus + self.adj) * self.std * self.ctxratio)
        
        self.denominator = 2.0 * (sqrt_term ** 2)
        
                                                        
        if self.isContext:
            size = len(self.parent.get_list()) if hasattr(self.parent, 'get_list') else 1
            self.ctxratio = trialLength / float(size)
            self.ratio = self.ctxratio
        else:
            self.ctxratio = 1.0
            self.ratio = 1.0
    
    def setPhase(self, phase: int):
        self.session = 1
        self.combination = 1
        self.trialTypeCount.clear()
        
                                                                    
        if phase > self.phase and hasattr(self, 'subelementWeights') and self.subelementWeights is not None:
            for i in range(len(self.subelementWeights)):
                for j in range(len(self.subelementWeights[0])):
                    for k in range(len(self.subelementWeights[0][0])):
                        self.oldElementWeights[i][j][k] = self.subelementWeights[i][j][k]
        
                                                                            
        if not self.isUS and not self.isContext and (self.random_midpoint is None or phase > self.phase):
                                                                 
                                                                                    
            trial_length = getattr(self.group, 'trial_length', 100)                        
            random_timepoint = self.rng.randint(0, trial_length)
            self.random_midpoint = random_timepoint / trial_length                          
            print(f"DEBUG: {self.getName()} random midpoint set to {self.random_midpoint:.3f} (timepoint {random_timepoint}/{trial_length}) for phase {phase}")
            print(f"DEBUG: RNG state: {self.rng.getstate()[1][0] if hasattr(self.rng, 'getstate') else 'unknown'}")
        elif self.getName() in ['A', 'B']:
            print(f"DEBUG: {self.getName()} NOT setting random midpoint - isUS={self.isUS}, isContext={self.isContext}, random_midpoint={self.random_midpoint}, phase={phase}, self.phase={self.phase}")
        
        self.phase = phase
        
                                                     
        if phase == 0:
            self.subelementWeights = np.zeros((int(self.subelementNumber), self.totalStimuli, self.totalMax), dtype=np.float32)
            self.oldElementWeights = np.zeros((int(self.subelementNumber), self.totalStimuli, self.totalMax), dtype=np.float32)
            self.kickedIn = False
            self.variableSalience[0] = self.alphaR
            self.csVariableSalience[0] = self.alphaN
            self.totalCSError[phase] = 0.0
            self.totalUSError[phase] = 0.0
            self.group.add_to_map("0", np.zeros((self.totalStimuli, self.totalMax), dtype=np.float32),
                                 self.elementCurrentWeightsKey, True)
        
                                                       
        if phase > 0:
            self.totalCSError[phase] = self.totalCSError[phase - 1]
            self.totalUSError[phase] = self.totalUSError[phase - 1]
            self.variableSalience[phase] = self.variableSalience[phase - 1]
            self.csVariableSalience[phase] = self.csVariableSalience[phase - 1]
            
                                                 
            for j in range(len(self.subelementWeights[0]) if self.subelementNumber != 0 else 0):
                for k in range(len(self.subelementWeights[0][0]) if self.subelementNumber != 0 else 0):
                    for i in range(len(self.subelementWeights)):
                        self.subelementWeights[i][j][k] = self.oldElementWeights[i][j][k]
            
                               
            temp = self.group.get_from_db(str(phase - 1), self.elementCurrentWeightsKey)
            if temp is not None:
                self.group.add_to_map(str(phase), temp, self.elementCurrentWeightsKey, True)
            else:
                temp = np.zeros((self.totalStimuli, self.totalMax), dtype=np.float32)
                self.group.add_to_map(str(phase), temp, self.elementCurrentWeightsKey, True)
            
                                     
            if phase - 2 >= 0:
                self.group.get_maps()[self.elementCurrentWeightsKey].pop(str(phase - 2), None)
    
    def reset(self, last: bool, currentTrials: int):
        self.activation = 0.0
        self.assoc = 0.0
        
                                           
        if self.phase > 0:
            self.totalCSError[self.phase] = self.totalCSError[self.phase - 1]
            self.totalUSError[self.phase] = self.totalUSError[self.phase - 1]
            self.variableSalience[self.phase] = self.variableSalience[self.phase - 1]
            self.csVariableSalience[self.phase] = self.csVariableSalience[self.phase - 1]
            
            for j in range(len(self.subelementWeights[0])):
                for k in range(len(self.subelementWeights[0][0])):
                    for i in range(len(self.subelementWeights)):
                        self.subelementWeights[i][j][k] = self.oldElementWeights[i][j][k]
            
            current = self.group.get_from_db(str(self.phase - 1), self.elementCurrentWeightsKey)
            if current is not None:
                self.group.add_to_map(str(self.phase), current, self.elementCurrentWeightsKey, True)
            else:
                current = np.zeros((self.totalStimuli, self.totalMax), dtype=np.float32)
                self.group.add_to_map(str(self.phase), current, self.elementCurrentWeightsKey, True)
            
            if self.phase - 2 >= 0:
                self.group.get_maps()[self.elementCurrentWeightsKey].pop(str(self.phase - 2), None)
        
        if self.phase == 0:
            self.subelementWeights = np.zeros((int(self.subelementNumber), self.totalStimuli, self.totalMax), dtype=np.float32)
            self.group.add_to_map("0", np.zeros((self.totalStimuli, self.totalMax), dtype=np.float32),
                                 self.elementCurrentWeightsKey, True)
        
        if self.subelementActivations is not None:
            self.subelementActivations.fill(0)
        
                             
        for i in range(self.totalStimuli):
            self.group.add_to_map(str(i), np.zeros(self.totalMax, dtype=np.float32),
                                 self.eligibilitiesKey, True)
        
        self.timepoint = 0
        self.directActivation = 0.0
        
        if self.group.get_phases()[self.phase].is_random() and self.phase == 0:
            self.variableSalience[self.phase] = self.alphaR
            self.csVariableSalience[self.phase] = self.alphaN
        
        self.wasActive = False
        self.firstPass = True
        self.isStored = False
        
        self.trialCount -= min(self.trialCount, currentTrials)
    
    def resetForNextTimepoint(self):
        if not self.hasReset:
            self.maxActivation = 0.0
            self.hasReset = True
            self.isStored = False
            self.firstPass = True
            self.activation = 0.0
            self.subelementActivations = np.zeros(int(self.subelementNumber), dtype=np.int32)
    
    def store(self):
        if not self.isStored:
            self.timepoint = 0
            self.directActivation = 0.0
            self.subelementActivations = np.zeros(int(self.subelementNumber), dtype=np.int32)
            self.trialCount += 1
            self.isStored = True
    
    def incrementTimepoint(self, time: int) -> str:
        self.hasReset = False
        return self.elementCurrentWeightsKey
    
    def setActive(self, name: str, b: bool, durationPoint: float):
        if b and not self.disabled:
            self.wasActive = True
        self.durationPoint = durationPoint
        
        if not self.isA:
            self.isA = False if self.a is None else self.a.get_has_been_active()
        if not self.isB:
            self.isB = False if self.b is None else self.b.get_has_been_active()
        
        self.setParams()
    
    def setParams(self):
        if "c" not in self.getName():
            return
        
        if self.isA and self.isB and self.notTotal:
            self.std = self.a.get_list()[0].getSTD() / 2.0 + self.b.get_list()[0].getSTD() / 2.0
            self.alphaR = self.a.getRAlpha() / 2.0 + self.b.getRAlpha() / 2.0
            self.alphaN = self.a.getNAlpha() / 2.0 + self.b.getNAlpha() / 2.0
            
            if self.trialCount == 0:
                self.variableSalience[0] = self.alphaR
                self.csVariableSalience[0] = self.alphaN
            
            self.variableSalience[self.phase] = self.alphaR
            self.csVariableSalience[self.phase] = self.alphaN
            self.salience = self.a.getSalience() / 2.0 + self.b.getSalience() / 2.0
            self.cscLikeness = self.a.getCSCLike() / 2.0 + self.b.getCSCLike() / 2.0
            
            self.averageUSError = max(0.0, self.alphaR)
            self.averageCSError = max(0.0, self.alphaN)
            self.notTotal = False
            self.kickedIn = True
        
        elif self.isA and not self.kickedIn:
            self.std = self.a.get_list()[0].getSTD()
            self.alphaR = self.a.getRAlpha()
            self.alphaN = self.a.getNAlpha()
            
            if self.trialCount == 0:
                self.variableSalience[0] = self.alphaR
                self.csVariableSalience[0] = self.alphaN
            
            self.averageUSError = max(0.0, self.alphaR)
            self.averageCSError = max(0.0, self.alphaN)
            self.salience = self.a.getSalience()
            self.cscLikeness = self.a.getCSCLike()
            self.kickedIn = True
        
        elif self.isB and not self.kickedIn:
            self.std = self.b.get_list()[0].getSTD()
            self.alphaR = self.b.getRAlpha()
            self.alphaN = self.b.getNAlpha()
            
            if self.trialCount == 0:
                self.variableSalience[0] = self.alphaR
                self.csVariableSalience[0] = self.alphaN
            
            self.averageUSError = max(0.0, self.alphaR)
            self.averageCSError = max(0.0, self.alphaN)
            self.salience = self.b.getSalience()
            self.cscLikeness = self.b.getCSCLike()
            self.kickedIn = True
    
    def initialize(self, a, b):
        self.a = a
        self.b = b
        self.setParams()
    
    def setVariableSalience(self, vs: float):
        iti = self._get_iti()
        temp_length = self.totalMax + iti
        temp = np.zeros(temp_length, dtype=np.float32)
        ob3 = self.group.get_from_db(str(self.trialCount), self.aggregateSaliencesKey)
        if ob3 is not None:
            temp = ob3
        if self.timepoint >= len(temp):
            new_temp = np.zeros(self.timepoint + 1, dtype=np.float32)
            new_temp[:len(temp)] = temp
            temp = new_temp
        temp[self.timepoint] = (temp[self.timepoint] * (self.combination - 1) + self.variableSalience[self.phase]) / self.combination
        self.group.add_to_map(str(self.trialCount), temp, self.aggregateSaliencesKey, True)
        
        self.variableSalience[self.phase] = vs if vs > 0.001 else 0.0
    
    def setCSVariableSalience(self, vs: float):
        iti = self._get_iti()
        temp_length = self.totalMax + iti
        temp = np.zeros(temp_length, dtype=np.float32)
        ob3 = self.group.get_from_db(str(self.trialCount), self.aggregateCSSaliencesKey)
        if ob3 is not None:
            temp = ob3
        if self.timepoint >= len(temp):
            new_temp = np.zeros(self.timepoint + 1, dtype=np.float32)
            new_temp[:len(temp)] = temp
            temp = new_temp
        temp[self.timepoint] = (temp[self.timepoint] * (self.combination - 1) + self.csVariableSalience[self.phase]) / self.combination
        self.group.add_to_map(str(self.trialCount), temp, self.aggregateCSSaliencesKey, True)
        
        self.csVariableSalience[self.phase] = vs if vs > 0.001 else 0.0
    
    def storeAverageUSError(self, d: float, act: float):
        self.oldUSError = self.averageUSError
        if self.averageUSError == 0:
            self.averageUSError = max(0.0, self.alphaR)
        else:
            self.averageUSError = self.averageUSError * (1.0 - act / self.factor) + d * act / self.factor
    
    def storeAverageCSError(self, d: float, act: float):
        self.oldCSError = self.averageCSError
        if self.averageCSError == 0:
            self.averageCSError = max(0.0, self.alphaN)
        else:
            self.averageCSError = self.averageCSError * (1.0 - act / self.factor) + d * act / self.factor
    
    def getTotalError(self, abs_error: float) -> float:
        self.totalUSError[self.phase] = 0.9997 * self.totalUSError[self.phase] + 0.0003 * abs_error
        return self.totalUSError[self.phase]
    
    def getTotalCSError(self, abs_error: float) -> float:
        self.totalCSError[self.phase] = 0.99997 * self.totalCSError[self.phase] + 0.00003 * abs_error
        return self.totalCSError[self.phase]
    
    def setCurrentTrialString(self, currentSeq: str):
        if currentSeq in self.trialTypeCount:
            self.trialTypeCount[currentSeq] = self.trialTypeCount[currentSeq] + 1
        else:
            self.trialTypeCount[currentSeq] = 1
        self.currentTrialString = currentSeq
    
    def setNextString(self, nextSeq: str):
        """Set next trial string"""
        self.nextTrialString = nextSeq
    
             
    def getName(self) -> str:
        return self.name
    
    def getMicroIndex(self) -> int:
        return self.microstimulusIndex
    
    def getDirectActivation(self) -> float:
                                                                                                        
        return self.directActivation
    
    def getGeneralActivation(self) -> float:
        return self.generalActivation
    
    def getAssoc(self) -> float:
        return self.assoc
    
    def getDurationPoint(self) -> float:
        return self.durationPoint
    
    def getAlpha(self) -> float:
        return self.beta * self.intensity if self.isUS else self.salience
    
    def getParent(self):
        return self.parent
    
    def getSTD(self) -> float:
        return self.std
    
    def getCurrentUSError(self) -> float:
        return self.averageUSError
    
    def getOldUSError(self) -> float:
        return self.oldUSError
    
    def getCurrentCSError(self) -> float:
        return self.averageCSError
    
    def getOldCSError(self) -> float:
        return self.oldCSError
    
    def getVariableSalience(self) -> float:
        return self.variableSalience[self.phase]
    
    def getCSVariableSalience(self) -> float:
        return self.csVariableSalience[self.phase]
    
    def isCommon(self) -> bool:
        return len(self.getName()) > 1
    
             
    def setRAlpha(self, alpha: float):
        self.alphaR = alpha
        self.variableSalience[0] = alpha
    
    def setNAlpha(self, alphaN: float):
        self.alphaN = alphaN
        self.csVariableSalience[0] = alphaN
    
    def setBeta(self, beta: float):
        self.beta = beta
    
    def setSalience(self, salience: float):
        self.salience = salience
    
    def setIntensity(self, intensity: float):
        self.intensity = intensity
    
    def setDisabled(self, disabled: bool):
        self.disabled = disabled
    
    def setVartheta(self, vartheta: float):
        if vartheta is None:
            print(f"WARNING: setVartheta called with None for {self.name}")
        self.vartheta = vartheta
    
    def setCSCLike(self, c: float):
        self.cscLikeness = c
    
    def setUSCV(self, cv: float):
        self.USCV = cv
    
    def setCSCV(self, cscv: float):
        self.std = cscv
    
    def setUSScalar(self, usScalar: float):
        self.USCV *= usScalar * usScalar
    
    def setCSScalar(self, csScalar: float):
        if self.std is None:
            self.std = 1.0                 
        if csScalar is None:
            csScalar = 1.0                 
        self.std *= csScalar * csScalar
    
    def setUSPersistence(self, p: float):
        self.usPersistence = p
    
    def setUSSalience(self, usS: float):
        pass                                 
    
                     
    def randomWithRange(self, min_val: int, max_val: int) -> int:
        range_val = (max_val - min_val) + 1
        return int(self.rng.random() * range_val) + min_val
    
    def _get_iti(self) -> int:
        iti = 0
        for sp in self.group.get_phases():
            iti = max(iti, int(sp.get_iti().get_minimum()))
        return iti
    
    def __str__(self) -> str:
        return f"{self.name} α({self.alphaR})"