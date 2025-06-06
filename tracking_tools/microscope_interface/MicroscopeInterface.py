import time
from ..logger.logger import init_logger
import pymcs

class MicroscopeInterface:
    def __init__(self) :
        self.microscope = pymcs.Microscope()
        self.connect()
        self.time_lapse_controller = pymcs.TimeLapseController(self.microscope)
        self.stage_xyz = pymcs.StageXYZ(self.microscope, "STAGE")

    def wait_for_pause(self, timeout_ms) :
        position_name, timepoint, timeout = self.time_lapse_controller.wait_for_pause(timeout_ms)
        return position_name, timepoint, timeout

    def pause_after_position(self) :
        self.time_lapse_controller.pause_after_position()

    def no_pause_after_position(self) :
        self.time_lapse_controller.no_pause_after_position()

    def continue_from_pause(self) :
        self.time_lapse_controller.continue_from_pause()

    def relative_move(self, position_name, shift_x, shift_y, shift_z) :
        pos = self.stage_xyz.position_get(position_name=position_name)
        self.stage_xyz.position_set(
            position_name=position_name,
            position_x=pos.position_x - shift_x,
            position_y=pos.position_y - shift_y,
            position_z=pos.position_z - shift_z,
        )

    def connect(self) :
        self.microscope.connect()

    def disconnect(self) :
        self.microscope.disconnect()


class SimulatedMicroscopeInterface :
    def __init__(self, position_names, max_timeout=8) :
        self.position_names = position_names
        self.nb_positions = len(position_names)
        self.current_position_index = 0
        self.timepoint = 0
        self.timeout_count = 0
        # Send 3 timeouts before pausing
        self.max_timeout = max_timeout
        # Set default logger
        self.logger = init_logger("SimulatedMicroscopeInterface")

    def wait_for_pause(self, timeout_ms) :
        time.sleep(timeout_ms/1000)
        # Send some timeouts before pausing
        self.timeout_count = (self.timeout_count + 1) % (self.max_timeout + 1)
        if self.timeout_count % self.max_timeout != 0 :
            self.logger.info("Sending timeout")
            return None, None, True
        # Go through positions in a round robin cycle
        current_pos = self.position_names[self.current_position_index]
        # Update timepoint if after a full cycle
        if self.current_position_index == 0 :
            self.timepoint = self.timepoint + 1
        # Update position for the next call
        self.current_position_index = (self.current_position_index + 1) % self.nb_positions
        self.logger.info("Pausing for a new position")
        return current_pos, self.timepoint, False
    
    def pause_after_position(self) :
        return
    
    def no_pause_after_position(self) :
        return
    
    def continue_from_pause(self) :
        return
    
    def relative_move(self, position_name, shift_x, shift_y, shift_z) :
        return
    
    def connect(self) :
        return
    
    def disconnect(self) :
        return
    
    
