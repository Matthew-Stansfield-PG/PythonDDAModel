import time
from threading import Lock
from typing import Optional


class ModelControl:
    
    def __init__(self):
        self.progress = 0
        self.total_progress = 0
        self.complete = False
        self.cancelled = False
        self.made_export = False
        
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.estimated_cycle_time = 0.0
        
        self._lock = Lock()
        
        self.progress_callback = None
        self.completion_callback = None
        
    def set_total_progress(self, total: int):
        with self._lock:
            self.total_progress = total
            
    def get_total_progress(self) -> int:
        with self._lock:
            return self.total_progress
            
    def increment_progress(self, amount: int = 1):
        with self._lock:
            self.progress += amount
            
            current_time = time.time()
            if self.progress > 0:
                elapsed = current_time - self.start_time
                self.estimated_cycle_time = (elapsed / self.progress) * 1000
                
            self.last_update_time = current_time
            
        if self.progress_callback:
            self.progress_callback(self.progress, self.total_progress)
            
    def get_progress(self) -> int:
        with self._lock:
            return self.progress
            
    def get_progress_percentage(self) -> float:
        with self._lock:
            if self.total_progress == 0:
                return 0.0
            return (self.progress / self.total_progress) * 100
            
    def set_complete(self, complete: bool):
        with self._lock:
            self.complete = complete
            
        if complete and self.completion_callback:
            self.completion_callback()
            
    def is_complete(self) -> bool:
        with self._lock:
            return self.complete
            
    def cancel(self):
        with self._lock:
            self.cancelled = True
            
    def is_cancelled(self) -> bool:
        with self._lock:
            return self.cancelled
            
    def reset(self):
        with self._lock:
            self.progress = 0
            self.total_progress = 0
            self.complete = False
            self.cancelled = False
            self.made_export = False
            self.start_time = time.time()
            self.last_update_time = time.time()
            self.estimated_cycle_time = 0.0
            
    def get_estimated_cycle_time(self) -> float:
        with self._lock:
            return self.estimated_cycle_time
            
    def get_estimated_remaining_time(self) -> float:
        with self._lock:
            if self.progress == 0 or self.total_progress == 0:
                return 0.0
                
            remaining_steps = self.total_progress - self.progress
            return (self.estimated_cycle_time * remaining_steps) / 1000
            
    def get_elapsed_time(self) -> float:
        return time.time() - self.start_time
        
    def set_made_export(self, made: bool):
        with self._lock:
            self.made_export = made
            
    def made_export(self) -> bool:
        with self._lock:
            return self.made_export
            
    def set_progress_callback(self, callback):
        self.progress_callback = callback
        
    def set_completion_callback(self, callback):
        self.completion_callback = callback
        
    def get_status_string(self) -> str:
        with self._lock:
            if self.cancelled:
                return "Cancelled"
            elif self.complete:
                return "Complete"
            elif self.progress == 0:
                return "Not started"
            else:
                percentage = self.get_progress_percentage()
                remaining = self.get_estimated_remaining_time()
                
                if remaining > 3600:
                    time_str = f"{remaining/3600:.1f} hours"
                elif remaining > 60:
                    time_str = f"{remaining/60:.1f} minutes"
                else:
                    time_str = f"{remaining:.0f} seconds"
                    
                return f"Running... {percentage:.1f}% complete, ~{time_str} remaining"