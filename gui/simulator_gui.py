import sys
from pathlib import Path
                                          
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout,
                                 QHBoxLayout, QPushButton, QTextEdit,
                                 QLabel, QSpinBox, QDoubleSpinBox,
                                 QGroupBox, QTabWidget, QTableWidget,
                                 QMessageBox, QFileDialog, QProgressDialog,
                                 QSplitter, QTreeWidget, QTreeWidgetItem,
                                 QSlider, QCheckBox, QComboBox, QLineEdit,
                                 QGridLayout, QScrollArea, QFrame, QApplication,
                                 QTableWidgetItem)
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
    from PyQt5.QtGui import QFont, QPalette, QColor
except ImportError:
    sys.exit(1)

import json
import numpy as np
import numbers
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import time
from typing import Dict, List, Any, Optional

from simulation.simModel import SimModel
from simulation.sim_group import SimGroup
from simulation.SimPhase import SimPhase
from core.stimulus import Stimulus
from core.SimElementExact import StimulusElement
from core.trial import Trial
from core.cs import CS


class WeightUpdateCanvas(FigureCanvas):

    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)

                      
        self.weight_data = {}
        self.activation_data = {}
        self.error_data = {}
        self.asymptote_data = {}

    def update_weights(self, stimulus_name: str, weights: Dict[str, float], trial: int):
        if stimulus_name not in self.weight_data:
            self.weight_data[stimulus_name] = {'trials': [], 'weights': {}}

        self.weight_data[stimulus_name]['trials'].append(trial)
        for target, weight in weights.items():
            if target not in self.weight_data[stimulus_name]['weights']:
                self.weight_data[stimulus_name]['weights'][target] = []
            self.weight_data[stimulus_name]['weights'][target].append(weight)

        self.plot_weights()

    def update_weight_display_from_model(self, model):
        if not hasattr(model, 'groups') or not model.groups:
            return

        for group_name, group in model.groups.items():
            if not hasattr(group, 'cues_map') or not group.cues_map:
                continue

            for stim_name, stimulus in group.cues_map.items():
                v_values = {}
                if hasattr(stimulus, 'names') and stimulus.names:
                    for target_name in stimulus.names:
                        if hasattr(stimulus, 'get_v_value'):
                            v_values[target_name] = stimulus.get_v_value(target_name)
                        elif hasattr(stimulus, 'average_weights') and stimulus.average_weights is not None:
                            if target_name in stimulus.names:
                                v_values[target_name] = stimulus.average_weights[stimulus.names.index(target_name)]

                                       
                if v_values:
                    trial_count = getattr(stimulus, 'trial_count', 0)
                    self.update_weights(stim_name, v_values, trial_count)

    def display_trial_weights(self, model, trial_type: int = 0, phase: int = 0):
        if not hasattr(model, 'groups') or not model.groups:
            return

                                  
        weight_data = []
        headers = ['Stimulus']

        for group_name, group in model.groups.items():
            if not hasattr(group, 'cues_map') or not group.cues_map:
                continue

                                                
            all_stimuli = set()
            for stim in group.cues_map.values():
                if hasattr(stim, 'names'):
                    all_stimuli.update(stim.names)
            headers.extend(sorted(all_stimuli))

                                                 
            for stim_name, stimulus in group.cues_map.items():
                if hasattr(stimulus, 'get_trial_average_weights'):
                    trial_weights = stimulus.get_trial_average_weights(trial_type, phase)
                    if trial_weights is not None and len(trial_weights) > 0:
                                                      
                        last_trial_weights = trial_weights[-1] if len(trial_weights) > 0 else np.zeros(len(headers)-1)

                        row = [stim_name]
                        for target in sorted(all_stimuli):
                            if target in stimulus.names:
                                idx = stimulus.names.index(target)
                                if idx < len(last_trial_weights):
                                    row.append(f"{last_trial_weights[idx]:.6f}")
                                else:
                                    row.append("0.000000")
                            else:
                                row.append("0.000000")
                        weight_data.append(row)

                                     
        self.update_weight_table(headers, weight_data)

    def update_weight_table(self, headers: List[str], data: List[List[str]]):
        """Update the weight table display."""
                                                               
                                                                         
        pass

    def update_activations(self, stimulus_name: str, activations: List[float], timepoints: List[int]):
        """Update activation display."""
        if stimulus_name not in self.activation_data:
            self.activation_data[stimulus_name] = {'timepoints': [], 'activations': []}

        self.activation_data[stimulus_name]['timepoints'].extend(timepoints)
        self.activation_data[stimulus_name]['activations'].extend(activations)

        self.plot_activations()

    def update_errors(self, stimulus_name: str, errors: List[float], trials: List[int]):
        """Update error display."""
        if stimulus_name not in self.error_data:
            self.error_data[stimulus_name] = {'trials': [], 'errors': []}

        self.error_data[stimulus_name]['trials'].extend(trials)
        self.error_data[stimulus_name]['errors'].extend(errors)

        self.plot_errors()

    def update_asymptotes(self, stimulus_name: str, asymptotes: List[float], trials: List[int]):
        """Update asymptote display."""
        if stimulus_name not in self.asymptote_data:
            self.asymptote_data[stimulus_name] = {'trials': [], 'asymptotes': []}

        self.asymptote_data[stimulus_name]['trials'].extend(trials)
        self.asymptote_data[stimulus_name]['asymptotes'].extend(asymptotes)

        self.plot_asymptotes()

    def plot_weights(self):
        """Plot weight evolution."""
        self.ax1.clear()
        self.ax1.set_title("Weight Evolution")
        self.ax1.set_xlabel("Trial")
        self.ax1.set_ylabel("Weight")
        self.ax1.grid(True, alpha=0.3)

        for stimulus_name, data in self.weight_data.items():
            for target, weights in data['weights'].items():
                if len(weights) > 0:
                    self.ax1.plot(data['trials'], weights,
                                label=f"{stimulus_name}→{target}", linewidth=2)

        self.ax1.legend()
        self.ax1.set_ylim(-1, 1)

    def plot_activations(self):
        """Plot activation curves."""
        self.ax2.clear()
        self.ax2.set_title("Activation Curves")
        self.ax2.set_xlabel("Time")
        self.ax2.set_ylabel("Activation")
        self.ax2.grid(True, alpha=0.3)

        for stimulus_name, data in self.activation_data.items():
            if len(data['activations']) > 0:
                self.ax2.plot(data['timepoints'], data['activations'],
                            label=stimulus_name, linewidth=2)

        self.ax2.legend()
        self.ax2.set_ylim(0, 1)

    def plot_errors(self):
        """Plot error curves."""
        self.ax3.clear()
        self.ax3.set_title("Prediction Error")
        self.ax3.set_xlabel("Trial")
        self.ax3.set_ylabel("Error")
        self.ax3.grid(True, alpha=0.3)

        for stimulus_name, data in self.error_data.items():
            if len(data['errors']) > 0:
                self.ax3.plot(data['trials'], data['errors'],
                            label=stimulus_name, linewidth=2)

        self.ax3.legend()

    def plot_asymptotes(self):
        """Plot asymptote evolution."""
        self.ax4.clear()
        self.ax4.set_title("Asymptote Evolution")
        self.ax4.set_xlabel("Trial")
        self.ax4.set_ylabel("Asymptote")
        self.ax4.grid(True, alpha=0.3)

        for stimulus_name, data in self.asymptote_data.items():
            if len(data['asymptotes']) > 0:
                self.ax4.plot(data['trials'], data['asymptotes'],
                            label=stimulus_name, linewidth=2)

        self.ax4.legend()
        self.ax4.set_ylim(0, 1)

        self.fig.tight_layout()
        self.draw()


class SimulationThread(QThread):
    """Thread for running simulations without blocking GUI."""

    progress_update = pyqtSignal(int, int)
    status_update = pyqtSignal(str)
    finished = pyqtSignal(dict)
    weight_update = pyqtSignal(str, dict, int)                                 

    def __init__(self, model, canvas):
        super().__init__()
        self.model = model
        self.canvas = canvas
        self.running = False

    def progress_callback(self, current, total):
        """Progress callback for enhanced simulation."""
        self.progress_update.emit(current, total)

    def run(self):
        """Run the simulation with real-time updates."""
        try:
            self.running = True
            self.status_update.emit("Initializing simulation...")

                                      
            if self.model.control:
                self.model.control.set_progress_callback(
                    lambda current, total: self.progress_update.emit(current, total)
                )

                                                            
            self.setup_weight_tracking()

            self.status_update.emit("Running simulation...")
            start_time = time.time()

                                      
            self.status_update.emit(f"Model has {len(self.model.groups)} groups")
            for group_name, group in self.model.groups.items():
                self.status_update.emit(f"Group {group_name} has {len(group.phases)} phases")
                for i, phase in enumerate(group.phases):
                    self.status_update.emit(f"  Phase {i} has {phase.trials} trials")

                                                     
            if hasattr(self.model, 'run_enhanced'):
                                                                    
                                                                                              
                self.model.run_enhanced(parallel=False, progress_callback=self.progress_callback)
            else:
                self.model.run()

                                                                  
            self.update_final_weights()

            end_time = time.time()
            self.status_update.emit(f"Simulation completed in {end_time - start_time:.2f} seconds")

                                   
            results = self.collect_results()

                                                     
            if hasattr(self.model, 'get_performance_stats'):
                results['performance_stats'] = self.model.get_performance_stats()
                results['gpu_info'] = self.model.get_gpu_info()

            self.finished.emit(results)

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.status_update.emit(f"Simulation failed: {str(e)}")
            self.status_update.emit(f"Error details: {error_details}")
            self.finished.emit({})
        finally:
            self.running = False

    def setup_weight_tracking(self):
        """Setup weight tracking for real-time updates."""
                                                                 
        for group in self.model.groups.values():
            for stimulus in group.cues_map.values():
                if hasattr(stimulus, 'update_element'):
                                           
                    original_update = stimulus.update_element

                    def make_tracking_update(stim_name, orig_method):
                        def tracking_update(*args, **kwargs):
                                                  
                            result = orig_method(*args, **kwargs)

                                                       
                            weights = {}
                            if hasattr(stimulus, 'get_weights'):
                                weights = stimulus.get_weights()
                            elif hasattr(stimulus, 'average_weights') and stimulus.average_weights is not None:
                                                             
                                weights = {f"weight_{i}": float(w) for i, w in enumerate(stimulus.average_weights)}

                            self.weight_update.emit(stim_name, weights, getattr(self.model, 'current_trial', 0))

                            return result
                        return tracking_update

                                                          
                    stimulus.update_element = make_tracking_update(stimulus.symbol, original_update)

    def update_final_weights(self):
        """Update weight display with final weights from simulation."""
        for group_name, group in self.model.groups.items():
            for stimulus_name, stimulus in group.cues_map.items():
                if hasattr(stimulus, 'average_weights') and stimulus.average_weights is not None:
                                                    
                    weights = {f"weight_{i}": float(w) for i, w in enumerate(stimulus.average_weights)}
                    self.weight_update.emit(stimulus_name, weights, 0)                                 

    def collect_results(self):
        """Collect simulation results."""
        results = {}

        for group_name, group in self.model.groups.items():
            group_results = {}

            for stimulus_name, stimulus in group.cues_map.items():
                if hasattr(stimulus, 'average_weights') and stimulus.average_weights is not None:
                    group_results[stimulus_name] = {
                        'weights': stimulus.average_weights.tolist(),
                        'weights_a': stimulus.average_weights_a.tolist() if hasattr(stimulus, 'average_weights_a') else [],
                        'final_prediction': stimulus.get_v_value("+") if "+" in stimulus.names else 0
                    }

            results[group_name] = group_results

        return results


class EnhancedDDASimulatorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.results = None
        self.simulation_thread = None
        self.weight_canvas = None

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("DDA Simulator - Python Version")
        self.setGeometry(100, 100, 1400, 900)

                               
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

                            
        main_layout = QHBoxLayout(central_widget)

                         
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

                               
        left_panel = self.create_control_panel()
        splitter.addWidget(left_panel)

                                                 
        right_panel = self.create_results_panel()
        splitter.addWidget(right_panel)

                                  
        splitter.setSizes([400, 1000])


                           
        self.status_label = QLabel("Ready")
        self.statusBar().addWidget(self.status_label)

                          
        self.model = None                                          

                                                       
        self.current_trial = 0

                                                         
                                                                     
        self.initialize_gpu_status()

    def create_control_panel(self):
        """Create the control panel."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

                                
        params_group = QGroupBox("Model Parameters")
        params_layout = QVBoxLayout(params_group)

                        
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("α(CS→US):"))
        self.alpha_r_spin = QDoubleSpinBox()
        self.alpha_r_spin.setRange(0.01, 2.0)
        self.alpha_r_spin.setSingleStep(0.01)
        self.alpha_r_spin.setValue(0.5)
        lr_layout.addWidget(self.alpha_r_spin)
        params_layout.addLayout(lr_layout)

        lr2_layout = QHBoxLayout()
        lr2_layout.addWidget(QLabel("α(CS→CS):"))
        self.alpha_n_spin = QDoubleSpinBox()
        self.alpha_n_spin.setRange(0.01, 2.0)
        self.alpha_n_spin.setSingleStep(0.01)
        self.alpha_n_spin.setValue(0.5)
        lr2_layout.addWidget(self.alpha_n_spin)
        params_layout.addLayout(lr2_layout)

                       
        us_layout = QHBoxLayout()
        us_layout.addWidget(QLabel("β+:"))
        self.beta_spin = QDoubleSpinBox()
        self.beta_spin.setRange(0.01, 1.0)
        self.beta_spin.setSingleStep(0.01)
        self.beta_spin.setValue(0.2)
        us_layout.addWidget(self.beta_spin)
        params_layout.addLayout(us_layout)

                  
        sal_layout = QHBoxLayout()
        sal_layout.addWidget(QLabel("CS Salience:"))
        self.salience_spin = QDoubleSpinBox()
        self.salience_spin.setRange(0.01, 2.0)
        self.salience_spin.setSingleStep(0.01)
        self.salience_spin.setValue(0.1)
        sal_layout.addWidget(self.salience_spin)
        params_layout.addLayout(sal_layout)

                            
        ctx_layout = QHBoxLayout()
        ctx_layout.addWidget(QLabel("Context Salience:"))
        self.ctx_salience_spin = QDoubleSpinBox()
        self.ctx_salience_spin.setRange(0.01, 1.0)
        self.ctx_salience_spin.setSingleStep(0.01)
        self.ctx_salience_spin.setValue(0.07)
        ctx_layout.addWidget(self.ctx_salience_spin)
        params_layout.addLayout(ctx_layout)

                  
        ts_layout = QHBoxLayout()
        ts_layout.addWidget(QLabel("Timestep Size:"))
        self.timestep_spin = QDoubleSpinBox()
        self.timestep_spin.setRange(0.01, 2.0)
        self.timestep_spin.setSingleStep(0.01)
        self.timestep_spin.setValue(1.0)
        ts_layout.addWidget(self.timestep_spin)
        params_layout.addLayout(ts_layout)

                        
        runs_layout = QHBoxLayout()
        runs_layout.addWidget(QLabel("Number of Runs:"))
        self.runs_spin = QSpinBox()
        self.runs_spin.setRange(1, 100)
        self.runs_spin.setValue(1)
        runs_layout.addWidget(self.runs_spin)
        params_layout.addLayout(runs_layout)

                 
        self.use_context_cb = QCheckBox("Use Context")
        self.use_context_cb.setChecked(True)
        params_layout.addWidget(self.use_context_cb)

        self.external_save_cb = QCheckBox("External Save")
        self.external_save_cb.setChecked(True)
        params_layout.addWidget(self.external_save_cb)

        layout.addWidget(params_group)

                                           
        gpu_group = QGroupBox("GPU & Parallel Processing")
        gpu_layout = QVBoxLayout()
        gpu_group.setLayout(gpu_layout)

                      
        gpu_controls = QHBoxLayout()
        self.gpu_enabled_cb = QCheckBox("Enable GPU Acceleration")
        self.gpu_enabled_cb.setChecked(True)                                                       
        self.gpu_enabled_cb.setToolTip("Use GPU acceleration for faster simulations")
        gpu_controls.addWidget(self.gpu_enabled_cb)

        self.parallel_enabled_cb = QCheckBox("Advanced Parallel Processing")
        self.parallel_enabled_cb.setChecked(False)                                              
        self.parallel_enabled_cb.setToolTip("Use multi-process parallel processing (may cause issues)")
        gpu_controls.addWidget(self.parallel_enabled_cb)

        self.memory_efficient_cb = QCheckBox("Memory Efficient Mode")
        self.memory_efficient_cb.setChecked(False)
        self.memory_efficient_cb.setToolTip("Use memory-efficient processing for large simulations")
        gpu_controls.addWidget(self.memory_efficient_cb)
        gpu_layout.addLayout(gpu_controls)

                           
        advanced_layout = QHBoxLayout()
        advanced_layout.addWidget(QLabel("Max Workers:"))
        self.max_workers_spin = QSpinBox()
        self.max_workers_spin.setRange(1, 16)
        self.max_workers_spin.setValue(4)
        self.max_workers_spin.setToolTip("Maximum number of parallel workers")
        advanced_layout.addWidget(self.max_workers_spin)

        advanced_layout.addWidget(QLabel("Batch Size:"))
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(64, 4096)
        self.batch_size_spin.setValue(1024)
        self.batch_size_spin.setSingleStep(64)
        self.batch_size_spin.setToolTip("Batch size for GPU operations")
        advanced_layout.addWidget(self.batch_size_spin)
        gpu_layout.addLayout(advanced_layout)

                    
        self.gpu_status_label = QLabel("GPU Status: Checking...")
        self.gpu_status_label.setStyleSheet("color: gray;")
        gpu_layout.addWidget(self.gpu_status_label)

        layout.addWidget(gpu_group)

                                 
        design_group = QGroupBox("Experiment Design")
        design_layout = QVBoxLayout(design_group)

                          
        self.design_text = QTextEdit()
        self.design_text.setMaximumHeight(200)
        self.design_text.setPlainText(
            "# Blocking Experiment Design\n"
            "Group: BlockingTest\n"
            "Phase1: A+ (5 trials)\n"
            "Phase2: AB+ (5 trials)\n"
            "Phase3: B- (1 trial)\n"
        )
        design_layout.addWidget(self.design_text)

                             
        parse_button = QPushButton("Parse Design")
        parse_button.clicked.connect(self.parse_design)
        design_layout.addWidget(parse_button)

        layout.addWidget(design_group)

                         
        button_widget = QWidget()
        button_layout = QVBoxLayout(button_widget)

        self.run_button = QPushButton("Run Simulation")
        self.run_button.clicked.connect(self.run_simulation)
        self.run_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        button_layout.addWidget(self.run_button)

        self.stop_button = QPushButton("Stop Simulation")
        self.stop_button.clicked.connect(self.stop_simulation)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
        button_layout.addWidget(self.stop_button)

        self.clear_button = QPushButton("Clear Results")
        self.clear_button.clicked.connect(self.clear_results)
        button_layout.addWidget(self.clear_button)

        layout.addWidget(button_widget)

        layout.addStretch()

        return widget

    def create_results_panel(self):
        """Create the results panel with visualization."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

                           
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)

                                  
        weight_tab = QWidget()
        weight_layout = QVBoxLayout(weight_tab)

        self.weight_canvas = WeightUpdateCanvas(weight_tab, width=12, height=8)
        weight_layout.addWidget(self.weight_canvas)

        tab_widget.addTab(weight_tab, "Weight Updates")

                                
        prediction_tab = QWidget()
        prediction_layout = QVBoxLayout(prediction_tab)
        self.prediction_canvas = FigureCanvas(Figure(figsize=(10, 6)))
        prediction_layout.addWidget(self.prediction_canvas)
        self.prediction_axes = self.prediction_canvas.figure.add_subplot(111)
        self.prediction_axes.set_title("Predictions Over Trials")
        self.prediction_axes.set_xlabel("Trial")
        self.prediction_axes.set_ylabel("Prediction (V)")
        self.prediction_axes.grid(True, alpha=0.3)
        tab_widget.addTab(prediction_tab, "Prediction History")

                           
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)

        self.results_table = QTableWidget()
        results_layout.addWidget(self.results_table)

        tab_widget.addTab(results_tab, "Results Table")

                                    
        performance_tab = QWidget()
        performance_layout = QVBoxLayout(performance_tab)

                                 
        stats_group = QGroupBox("Performance Statistics")
        stats_layout = QVBoxLayout(stats_group)

        self.performance_text = QTextEdit()
        self.performance_text.setReadOnly(True)
        self.performance_text.setMaximumHeight(200)
        stats_layout.addWidget(self.performance_text)

                        
        gpu_group = QGroupBox("GPU Information")
        gpu_layout = QVBoxLayout(gpu_group)

        self.gpu_info_text = QTextEdit()
        self.gpu_info_text.setReadOnly(True)
        self.gpu_info_text.setMaximumHeight(150)
        gpu_layout.addWidget(self.gpu_info_text)

                                 
        plots_group = QGroupBox("Performance Plots")
        plots_layout = QVBoxLayout(plots_group)

        self.performance_canvas = self.create_performance_plots()
        plots_layout.addWidget(self.performance_canvas)

        performance_layout.addWidget(stats_group)
        performance_layout.addWidget(gpu_group)
        performance_layout.addWidget(plots_group)

        tab_widget.addTab(performance_tab, "Performance")

                 
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)

        tab_widget.addTab(log_tab, "Simulation Log")

        return widget
    
    def parse_design(self):
        """Parse the experiment design from text and create proper DDA model structure."""
        try:
            design_text = self.design_text.toPlainText()
            self.log_message("Parsing experiment design...")

                                                    
            if self.model is None:
                self.model = self.create_enhanced_model()
                self.log_message("Created DDA model for design parsing")

                                               
            lines = design_text.strip().split('\n')
            current_group = None
            current_phase_num = 0
            phases_data = []                    

            self.log_message(f"Parsing {len(lines)} lines from design text")
            for i, line in enumerate(lines):
                line = line.strip()
                self.log_message(f"Line {i+1}: '{line}'")
                if line.startswith('Group:'):
                    group_name = line.split(':', 1)[1].strip()
                    current_group = self.model.add_group(group_name)
                    current_group.no_of_phases = 3                            
                    current_group._shared_rng = self.model._shared_rng
                    self.log_message(f"Created group: {group_name}")

                elif line.startswith('Phase'):
                    if ':' in line:
                        phase_name = line.split(':', 1)[1].strip()
                        current_phase_num += 1
                        
                                                                                       
                        trial_part = line.split(':', 1)[1].strip()
                                                                      
                        trial_types = [t.strip() for t in trial_part.split(',')]
                        
                        phase_trials = []
                        for trial_type in trial_types:
                            if '(' in trial_type and ')' in trial_type:
                                trial_name = trial_type.split('(')[0].strip()
                                count_part = trial_type.split('(')[1].split(')')[0].strip()

                                if 'trial' in count_part:
                                    count = int(count_part.split()[0])
                                                                                                   
                                    if trial_name != 'Ω':
                                                              
                                        for i in range(count):
                                            trial = Trial(trial_name, trial_number=i+1, group=current_group)
                                            phase_trials.append(trial)
                                        self.log_message(f"Added trial: {trial_name} x{count}")
                                    else:
                                        self.log_message(f"Skipped context (Ω) - always present")
                                else:
                                    self.log_message(f"Warning: Could not parse trial count from '{count_part}'")
                            else:
                                self.log_message(f"Warning: Could not parse trial type from '{trial_type}'")
                        
                        phases_data.append({
                            'phase_num': current_phase_num,
                            'trials': phase_trials,
                            'trial_type': trial_types[0] if trial_types else "Unknown"
                        })
                        self.log_message(f"Created phase {current_phase_num}: {phase_name} with {len(phase_trials)} trials")

                elif line and not line.startswith('#'):
                                                                          
                    if '(' in line and ')' in line:
                        trial_part = line.split('(')[0].strip()
                        count_part = line.split('(')[1].split(')')[0].strip()

                        if 'trial' in count_part:
                            count = int(count_part.split()[0])
                                                                                           
                            if trial_part != 'Ω':
                                                                        
                                if not phases_data:
                                    phases_data.append({'phase_num': 1, 'trials': [], 'trial_type': trial_part})
                                
                                for i in range(count):
                                    trial = Trial(trial_part, trial_number=i+1, group=current_group)
                                    phases_data[-1]['trials'].append(trial)
                                self.log_message(f"Added trial: {trial_part} x{count}")
                            else:
                                self.log_message(f"Skipped context (Ω) - always present")

                                                    
            if current_group and phases_data:
                current_group.no_of_phases = len(phases_data)
                
                                                     
                from simulation.configurables import TimingConfiguration, ITIConfig, ContextConfig
                
                timing = TimingConfiguration()
                iti = ITIConfig()
                context_cfg = ContextConfig()
                
                                                        
                for phase_data in phases_data:
                                                           
                    trial_counts = {}
                    for trial in phase_data['trials']:
                        trial_type = trial.trial_string
                        trial_counts[trial_type] = trial_counts.get(trial_type, 0) + 1
                    
                                                               
                    seq_parts = []
                    for trial_type, count in trial_counts.items():
                        seq_parts.append(f"{count}{trial_type}")
                    seq_string = "/".join(seq_parts)
                    
                    phase = SimPhase(
                        phase_num=phase_data['phase_num'],
                        sessions=1,
                        seq=seq_string,
                        order=phase_data['trials'],
                        stimuli2={"A": 0.5, "B": 0.5, "+": 1.0, "Context": 0.1},
                        sg=current_group,
                        random_order=False,
                        timing=timing,
                        iti=iti,
                        context=context_cfg,
                        trials_in_all_phases=len(phase_data['trials']),
                        listed_stimuli=["A", "B", "+", "Context"],
                        varying_vartheta=False
                    )
                    
                                                                         
                    phase.ordered_seq = phase_data['trials']
                    
                                    
                    phase.lambdaPlus = 1.0
                    phase.betaPlus = 0.1
                    phase.vartheta = 0.5
                    
                                          
                    if not hasattr(current_group, 'phases'):
                        current_group.phases = []
                    current_group.phases.append(phase)
                    
                    self.log_message(f"Created SimPhase {phase_data['phase_num']} with {len(phase_data['trials'])} trials")

                                                           
            self.log_message(f"Model type: {type(self.model)}")
            self.log_message(f"Model has initialize method: {hasattr(self.model, 'initialize')}")
            if hasattr(self.model, 'initialize'):
                self.model.initialize()
                self.log_message("Model initialized successfully")
            else:
                self.log_message("WARNING: Model does not have initialize method")

            self.log_message("Design parsing completed successfully")
            self.status_label.setText("Design parsed successfully")

        except Exception as e:
            self.log_message(f"Design parsing error: {str(e)}")
            import traceback
            self.log_message(f"Error details: {traceback.format_exc()}")
            QMessageBox.critical(self, "Design Error", str(e))

    def create_enhanced_model(self):
        """Create a model using the actual DDA simulation structure."""
        try:
                                                           
            from simulation.simModel import SimModel
            model = SimModel()
            model.use_context = True
            model.restrict_predictions = False
            model.context_alpha_r = 0.25
            model.context_alpha_n = 0.2
            model.context_salience = 0.07
            
                                                                   
            import random
            current_seed = 647926                                       
            model._shared_rng = random.Random(current_seed)
            
                                                    
            if hasattr(self, 'gpu_status_label'):
                self.gpu_status_label.setText("GPU: Using CPU (DDA Model)")
                self.gpu_status_label.setStyleSheet("color: blue;")
                self.log_message("Using actual DDA model with CPU processing")

            return model

        except Exception as e:
            self.log_message(f"Error creating DDA model: {e}")
            import traceback
            self.log_message(f"Error details: {traceback.format_exc()}")
                                      
            from simulation.simModel import SimModel
            return SimModel()

    def check_gpu_availability(self):
        """Check if GPU is available."""
        try:
            import cupy as cp
                                                
            test_array = cp.array([1, 2, 3])
            return True
        except ImportError:
            return False
        except Exception:
            return False

    def run_simulation(self):
        """Run the simulation using the actual DDA model."""
        try:
            if not self.model:
                self.model = self.create_enhanced_model()
                self.log_message("Created DDA model for simulation")

                                                         
            self.model.timestep_size = self.timestep_spin.value()
            self.model.set_combination_no(self.runs_spin.value())
            self.model.use_context = self.use_context_cb.isChecked()
            self.model.is_external_save = self.external_save_cb.isChecked()
            
                                     
            self.model.alpha_r = self.alpha_r_spin.value()
            self.model.alpha_n = self.alpha_n_spin.value()
            self.model.context_alpha_r = self.alpha_r_spin.value()                              
            self.model.context_alpha_n = self.alpha_n_spin.value()                              
            self.model.context_salience = self.ctx_salience_spin.value()

            self.log_message(f"Model has {len(self.model.groups)} groups")
            
                                           
            for group_name, group in self.model.groups.items():
                self.log_message(f"Running group: {group_name}")
                
                if hasattr(group, 'phases') and group.phases:
                    for phase in group.phases:
                                              
                        if hasattr(phase, 'betaPlus'):
                            phase.betaPlus = self.beta_spin.value()
                        if hasattr(phase, 'lambdaPlus'):
                            phase.lambdaPlus = 1.0                                  
                        if hasattr(phase, 'vartheta'):
                            phase.vartheta = 0.5
                        
                        self.log_message(f"Running phase {getattr(phase, 'phase_num', 'unknown')} with {len(getattr(phase, 'order', []))} trials")
                        
                                                                       
                        try:
                            if hasattr(phase, 'runSimulator'):
                                phase.runSimulator()
                                self.log_message(f"Phase {getattr(phase, 'phase_num', 'unknown')} completed successfully")
                            else:
                                self.log_message(f"Phase {getattr(phase, 'phase_num', 'unknown')} has no runSimulator method")
                        except Exception as e:
                            self.log_message(f"Error in phase {getattr(phase, 'phase_num', 'unknown')}: {str(e)}")
                            import traceback
                            self.log_message(f"Error details: {traceback.format_exc()}")
                else:
                    self.log_message(f"Group {group_name} has no phases or phases is empty")

                                                
            results = self.collect_dda_results()
            
                                         
            self.simulation_finished(results)
            
            self.log_message("DDA simulation completed successfully")
            self.status_label.setText("Simulation completed")
            
        except Exception as e:
            self.log_message(f"Simulation failed: {str(e)}")
            import traceback
            self.log_message(f"Error details: {traceback.format_exc()}")
            QMessageBox.critical(self, "Simulation Error", f"Simulation failed: {str(e)}")

    def collect_dda_results(self):
        """Collect results from the DDA model."""
        results = {}
        
        for group_name, group in self.model.groups.items():
            group_results = {}
            
            if hasattr(group, 'cues_map'):
                for stimulus_name, stimulus in group.cues_map.items():
                    if hasattr(stimulus, 'average_weights_a') and stimulus.average_weights_a is not None:
                                           
                        final_weights = stimulus.average_weights_a.tolist()

                                                                                
                        predictions = {}
                        if hasattr(group, 'last_predictions'):
                            predictions = group.last_predictions.get(stimulus_name, {})

                        group_results[stimulus_name] = {
                            'weights': final_weights,
                            'predictions': predictions,
                            'final_prediction': predictions.get('+', 0.0)
                        }

                        self.log_message(f"Stimulus {stimulus_name} final weights: {final_weights}")
                        self.log_message(f"Stimulus {stimulus_name} predictions (live V): {predictions}")
            
            results[group_name] = group_results

        if hasattr(self.model, 'prediction_history'):
            results['prediction_history'] = self.model.prediction_history
        
        return results

    def initialize_gpu_status(self):
        """Initialize GPU status display."""
        try:
                                              
            if not hasattr(self, 'gpu_status_label'):
                return

                                                                      
            gpu_available = self.check_gpu_availability()

            if gpu_available:
                                                                   
                temp_model = self.create_enhanced_model()
                if hasattr(temp_model, 'get_gpu_info'):
                    gpu_info = temp_model.get_gpu_info()
                else:
                    gpu_info = {'available': False, 'message': 'No GPU info method'}

                if gpu_info.get('available', False):
                    memory_total = gpu_info.get('memory_total', 0)
                    self.gpu_status_label.setText(f"GPU: Available ({memory_total:.1f} GB)")
                    self.gpu_status_label.setStyleSheet("color: green;")
                    self.log_message(f"GPU acceleration ready: {memory_total:.1f} GB VRAM")
                else:
                    self.gpu_status_label.setText("GPU: Not Available")
                    self.gpu_status_label.setStyleSheet("color: red;")
                    self.log_message(f"GPU not available: {gpu_info.get('message', 'Unknown error')}")
            else:
                self.gpu_status_label.setText("GPU: Not Available")
                self.gpu_status_label.setStyleSheet("color: red;")
                self.log_message("GPU not available: CuPy not installed or CUDA not available")

        except Exception as e:
            if hasattr(self, 'gpu_status_label'):
                self.gpu_status_label.setText(f"GPU: Error - {str(e)}")
                self.gpu_status_label.setStyleSheet("color: red;")
            self.log_message(f"GPU status check failed: {e}")

    def stop_simulation(self):
        """Stop the running simulation."""
        if self.simulation_thread and self.simulation_thread.isRunning():
            self.simulation_thread.running = False
            self.simulation_thread.terminate()
            self.simulation_thread.wait()

        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.log_message("Simulation stopped")

    def plot_prediction_history(self, history, stimuli_of_interest=None, targets_of_interest=None):
        self.prediction_axes.clear()
        self.prediction_axes.set_title("Predictions Over Trials")
        self.prediction_axes.set_xlabel("Trial")
        self.prediction_axes.set_ylabel("Prediction (V)")
        self.prediction_axes.grid(True, alpha=0.3)

        if history is None or len(history) == 0:
            self.prediction_canvas.draw()
            self.log_message("Prediction history is empty or unavailable")
            return

        if stimuli_of_interest is None:
            stimuli_of_interest = set()
            for entry in history:
                preds = entry.get('predictions', {})
                stimuli_of_interest.update(preds.keys())
            stimuli_of_interest = sorted(list(stimuli_of_interest))[:4]

        for stim_name in stimuli_of_interest:
            trials = []
            target_series: Dict[str, List[float]] = {}
            for entry in history:
                trial_idx = entry.get('global_trial', entry.get('trial'))
                preds = entry.get('predictions', {})
                stim_preds = preds.get(stim_name, {})
                if trial_idx is not None:
                    trials.append(trial_idx)
                    if targets_of_interest is None:
                        targets_to_plot = ['+'] if '+' in stim_preds else []
                    else:
                        targets_to_plot = targets_of_interest

                    for target in targets_to_plot:
                        value = stim_preds.get(target)
                        if value is None:
                            continue

                        if isinstance(value, dict):
                            value = value.get('total') or value.get('value') or next(iter(value.values()), 0.0)

                        if not isinstance(value, numbers.Number):
                            continue
                        target_series.setdefault(target, []).append(float(value))

            for target, values in target_series.items():
                if len(values) != len(trials):
                    min_len = min(len(values), len(trials))
                    values = values[:min_len]
                    t_arr = trials[:min_len]
                else:
                    t_arr = trials

                label = f"{stim_name}->{target}"
                self.prediction_axes.plot(t_arr, values, marker='o', label=label)

        if self.prediction_axes.lines:
            self.prediction_axes.legend()
        self.prediction_canvas.draw()
        self.log_prediction_history(history, stimuli_of_interest, targets_of_interest)

    def log_prediction_history(self, history, stimuli_of_interest=None, targets_of_interest=None):
        if history is None:
            self.log_message("No prediction history to log")
            return

        self.log_message("Prediction history summary:")
        for entry in history:
            trial_idx = entry.get('trial')
            predictions = entry.get('predictions', {})
            self.log_message(f"  Trial {trial_idx} predictions:")
            for stim, tgt_dict in predictions.items():
                if stim == '__element_details__':
                    continue
                if stimuli_of_interest and stim not in stimuli_of_interest:
                    continue
                for target, value in tgt_dict.items():
                    if target != '+' and (not targets_of_interest or '+' not in targets_of_interest):
                        continue
                    if isinstance(value, dict):
                        value = value.get('total') or value.get('value') or next(iter(value.values()), 0.0)
                    if targets_of_interest and target not in targets_of_interest:
                        continue
                    self.log_message(f"    {stim} -> {target}: {value}")

            detail_map = entry.get('element_predictions') or entry.get('details')
            if detail_map:
                self.log_message("    Element-level contributions:")
                for stim_name, elements in detail_map.items():
                    if stimuli_of_interest and stim_name not in stimuli_of_interest:
                        continue
                    for element_idx, contribs in elements.items():
                        if not contribs:
                            continue
                        contrib_strings = [f"{src}->{stim_name}: {val}" for src, val in contribs.items() if not targets_of_interest or src in targets_of_interest]
                        if contrib_strings:
                            self.log_message(f"      Element {stim_name}[{element_idx}]: {', '.join(contrib_strings)}")

    def simulation_finished(self, results):
        self.results = results
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        history = None
        if hasattr(self.model, 'prediction_history') and self.model.prediction_history:
            history = self.model.prediction_history
        elif 'prediction_history' in results:
            history = results['prediction_history']
        else:
            for group in self.model.get_groups().values():
                if hasattr(group, 'prediction_history') and group.prediction_history:
                    history = group.prediction_history
                    break

        if history:
            self.plot_prediction_history(history)
        else:
            self.log_message("No prediction history captured.")

        self.update_results_table()
        self.update_performance_display(results)
        self.log_message("Simulation completed successfully")
        self.status_label.setText("Simulation completed")

    def update_performance_display(self, results):
        """Update performance monitoring display."""
        if not hasattr(self, 'performance_text') or not hasattr(self, 'gpu_info_text'):
            return

                                       
        if 'performance_stats' in results:
            stats = results['performance_stats']
            perf_text = f"""Performance Statistics:
Total Time: {stats.get('total_time', 0):.2f} seconds
GPU Time: {stats.get('gpu_time', 0):.2f} seconds
Parallel Time: {stats.get('parallel_time', 0):.2f} seconds
Groups Processed: {stats.get('groups_processed', 0)}
GPU Weight Updates: {stats.get('weight_updates_gpu', 0)}
CPU Weight Updates: {stats.get('weight_updates_cpu', 0)}
Memory Peak: {stats.get('memory_peak', 0):.2f} GB"""
            self.performance_text.setText(perf_text)

                                
        if 'gpu_info' in results:
            gpu_info = results['gpu_info']
            if gpu_info.get('available', False):
                gpu_text = f"""GPU Information:
Status: Available
Device ID: {gpu_info.get('device_id', 'Unknown')}
Total Memory: {gpu_info.get('memory_total', 0):.2f} GB
Used Memory: {gpu_info.get('memory_used', 0):.2f} GB
Free Memory: {gpu_info.get('memory_free', 0):.2f} GB
CuPy Available: {gpu_info.get('cupy_available', False)}
CUDA Available: {gpu_info.get('cuda_available', False)}"""
            else:
                gpu_text = f"""GPU Information:
Status: Not Available
Reason: {gpu_info.get('message', 'Unknown error')}
CuPy Available: {gpu_info.get('cupy_available', False)}
CUDA Available: {gpu_info.get('cuda_available', False)}"""
            self.gpu_info_text.setText(gpu_text)

                                  
        self.update_performance_plots(results)

    def update_performance_plots(self, results):
        """Update performance monitoring plots."""
        if not hasattr(self, 'perf_ax1'):
            return

                              
        self.perf_ax1.clear()
        self.perf_ax2.clear()
        self.perf_ax3.clear()
        self.perf_ax4.clear()

                                       
        if 'performance_stats' in results:
            stats = results['performance_stats']
            total_time = stats.get('total_time', 0)
            gpu_time = stats.get('gpu_time', 0)
            parallel_time = stats.get('parallel_time', 0)
            cpu_time = total_time - gpu_time - parallel_time

            times = [cpu_time, gpu_time, parallel_time]
            labels = ['CPU', 'GPU', 'Parallel']
            colors = ['blue', 'green', 'orange']

            self.perf_ax1.bar(labels, times, color=colors)
            self.perf_ax1.set_title("Execution Time Breakdown")
            self.perf_ax1.set_ylabel("Time (seconds)")

                                  
            if total_time > 0:
                gpu_util = (gpu_time / total_time) * 100
                self.perf_ax2.bar(['GPU Utilization'], [gpu_util], color='green')
                self.perf_ax2.set_title("GPU Utilization")
                self.perf_ax2.set_ylabel("Percentage (%)")
                self.perf_ax2.set_ylim(0, 100)

                               
            memory_peak = stats.get('memory_peak', 0)
            if memory_peak > 0:
                self.perf_ax3.bar(['Peak Memory'], [memory_peak], color='red')
                self.perf_ax3.set_title("Memory Usage")
                self.perf_ax3.set_ylabel("Memory (GB)")

                             
            groups_processed = stats.get('groups_processed', 0)
            if total_time > 0 and groups_processed > 0:
                throughput = groups_processed / total_time
                self.perf_ax4.bar(['Throughput'], [throughput], color='purple')
                self.perf_ax4.set_title("Processing Throughput")
                self.perf_ax4.set_ylabel("Groups/Second")

                       
        self.performance_canvas.draw()

    def update_progress(self, current, total):
        """Update progress display."""
        if total > 0:
            progress = int((current / total) * 100)
            self.status_label.setText(f"Progress: {progress}% ({current}/{total})")

    def update_status(self, message):
        """Update status message."""
        self.status_label.setText(message)
        self.log_message(message)

    def update_weight_display(self, stimulus_name, weights, trial):
        """Update weight display in real-time."""
        if self.weight_canvas and weights:
            self.weight_canvas.update_weights(stimulus_name, weights, trial)
            self.log_message(f"Updated weights for {stimulus_name} at trial {trial}")

    def update_results_table(self):
        """Update the results table with DDA model data showing weights."""
        if not self.results:
            return

                                                               
        table_data = []

                                                                                             
        for group_name, group_combinations in self.results.items():
            if isinstance(group_combinations, list):
                                                            
                for combo_idx, combination in enumerate(group_combinations):
                    if isinstance(combination, dict) and 'phase_results' in combination:
                        for phase_idx, phase in enumerate(combination['phase_results']):
                            if isinstance(phase, dict) and 'trial_results' in phase:
                                for trial in phase['trial_results']:
                                    if isinstance(trial, dict):
                                                            
                                        trial_num = trial.get('trial_num', 0)
                                        trial_type = trial.get('trial_type', 'Unknown')
                                        is_probe = trial.get('is_probe', False)

                                                                           
                                        predictions = trial.get('predictions', {})
                                        for stimulus_name, prediction in predictions.items():
                                            table_data.append([
                                                group_name,
                                                f"Trial {trial_num}",
                                                trial_type,
                                                stimulus_name,
                                                f"{prediction:.4f}",
                                                "Probe" if is_probe else "Training"
                                            ])
            elif isinstance(group_combinations, dict):
                                                    
                if 'gpu_processed' in group_combinations:
                    table_data.append([
                        group_name,
                        "GPU Processing",
                        "Enhanced",
                        "GPU Accelerated",
                        f"{group_combinations.get('gpu_memory_used', 0):.1f} MB",
                        "GPU"
                    ])
                else:
                                                  
                    for key, value in group_combinations.items():
                        if isinstance(value, dict) and 'final_prediction' in value:
                            table_data.append([
                                group_name,
                                key,
                                "Final",
                                "Prediction",
                                f"{value['final_prediction']:.4f}",
                                "Standard"
                            ])

                                                                              
        if hasattr(self, 'model') and self.model and hasattr(self.model, 'groups'):
            for group_name, group in self.model.groups.items():
                if hasattr(group, 'cues_map'):
                                                                
                    stimulus_names = list(group.cues_map.keys())

                    for stimulus_name, stimulus in group.cues_map.items():
                        if hasattr(stimulus, 'get_v_value'):
                                                                                              
                            for target_name in stimulus_names:
                                if target_name != stimulus_name:                               
                                    v_value = stimulus.get_v_value(target_name)
                                    if abs(v_value) > 0.001:                                 
                                        table_data.append([
                                            group_name,
                                            "Final",
                                            "V-Value",
                                            f"{stimulus_name}→{target_name}",
                                            f"{v_value:.4f}",
                                            "Associative Strength"
                                        ])

                      
        self.results_table.setRowCount(len(table_data))
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels([
            "Group", "Trial", "Type", "Connection", "Weight", "Mode"
        ])

                        
        for row, data in enumerate(table_data):
            for col, value in enumerate(data):
                self.results_table.setItem(row, col, QTableWidgetItem(str(value)))

                                       
        self.results_table.resizeColumnsToContents()

    def clear_results(self):
        """Clear all results and visualizations."""
        self.results = None
        self.results_table.setRowCount(0)
        self.weight_canvas.weight_data.clear()
        self.weight_canvas.activation_data.clear()
        self.weight_canvas.error_data.clear()
        self.weight_canvas.asymptote_data.clear()
        self.weight_canvas.setup_plots()
        self.log_text.clear()
        self.log_message("Results cleared")

    def create_performance_plots(self):
        """Create performance monitoring plots."""
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        fig = Figure(figsize=(10, 6))
        canvas = FigureCanvas(fig)

                         
        self.perf_ax1 = fig.add_subplot(221)                  
        self.perf_ax2 = fig.add_subplot(222)                   
        self.perf_ax3 = fig.add_subplot(223)                
        self.perf_ax4 = fig.add_subplot(224)              

                     
        self.perf_ax1.set_title("Execution Time")
        self.perf_ax1.set_ylabel("Time (seconds)")
        self.perf_ax1.grid(True, alpha=0.3)

        self.perf_ax2.set_title("GPU Utilization")
        self.perf_ax2.set_ylabel("GPU Time (%)")
        self.perf_ax2.grid(True, alpha=0.3)

        self.perf_ax3.set_title("Memory Usage")
        self.perf_ax3.set_ylabel("Memory (GB)")
        self.perf_ax3.grid(True, alpha=0.3)

        self.perf_ax4.set_title("Throughput")
        self.perf_ax4.set_ylabel("Trials/Second")
        self.perf_ax4.grid(True, alpha=0.3)

        fig.tight_layout()
        return canvas

    def log_message(self, message):
        """Add message to log."""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def new_experiment(self):
        """Create a new experiment."""
        self.model = None                                          
        self.current_trial = 0
        self.clear_results()
        self.log_message("New experiment created")

                               
        if hasattr(self, 'design_text'):
            self.design_text.clear()

    def load_config(self, config_path):
        """Load configuration from file."""
        try:
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                self.log_message(f"Loaded configuration: {Path(config_path).name}")
                self.status_label.setText(f"Loaded: {Path(config_path).name}")
                return True
        except Exception as e:
            self.log_message(f"Failed to load config: {str(e)}")
        return False

    def load_experiment(self):
        """Load an experiment from file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Experiment", "", "JSON Files (*.json)"
        )

        if filename:
            try:
                self.model = None                                          
                self.model.current_trial = 0
                                                 
                if hasattr(self.model, 'load_config'):
                    self.model.load_config(filename)
                self.log_message(f"Loaded experiment: {Path(filename).name}")
                self.status_label.setText(f"Loaded: {Path(filename).name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load: {str(e)}")

    def save_experiment(self):
        """Save the current experiment."""
        if not self.model:
            QMessageBox.warning(self, "Warning", "No experiment to save")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Experiment", "", "JSON Files (*.json)"
        )

        if filename:
            try:
                config = self.model.get_config() if hasattr(self.model, 'get_config') else {}
                with open(filename, 'w') as f:
                    json.dump(config, f, indent=2)
                self.log_message(f"Saved experiment: {Path(filename).name}")
                self.status_label.setText(f"Saved: {Path(filename).name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save: {str(e)}")

    def export_results(self):
        """Export results to file."""
        if not self.results:
            QMessageBox.warning(self, "Warning", "No results to export")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "", "JSON Files (*.json)"
        )

        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(self.results, f, indent=2)
                self.log_message(f"Exported results: {Path(filename).name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export: {str(e)}")



def main():
    """Main function to run the GUI."""
    app = QApplication(sys.argv)
    
                                
    app.setApplicationName("Enhanced DDA Simulator")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("DDA Research")
    
                         
    gui = EnhancedDDASimulatorGUI()
    gui.show()
    
                     
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()






















