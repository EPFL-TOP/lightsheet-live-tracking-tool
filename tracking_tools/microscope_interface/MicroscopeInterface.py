import time
import os
import tifffile
from ..logger.logger import init_logger

class MicroscopeInterface_LS1:
    def __init__(self, positions_config) :
        import pymcs

        self.positions_config = positions_config
        # Get the position names seperated from the settings
        self.position_names = [posSetting.split("_")[0] for posSetting in self.positions_config.keys()]
        # Make the position to PositionSettings and channel lookup table
        self.pos_to_PosSettings = {}
        self.pos_to_Channel = {}
        for pos_name in positions_config.keys() :
            position_setting = pos_name.split("_")
            self.pos_to_PosSettings[position_setting[0]] = pos_name
            self.pos_to_Channel[position_setting[0]] = positions_config[pos_name]["filename"].replace(".tif","").split("_")[-1]

        self.microscope = pymcs.Microscope()
        self.connect()
        self.time_lapse_controller = pymcs.TimeLapseController(self.microscope)
        self.stage_xyz = pymcs.StageXYZ(self.microscope, "STAGE")

    # Waits for a new image
    def wait_for_image(self, timeout_ms) :
        timeout = False
        while not self.stop_requested :
            
            if not timeout:
                self.logger.info(f"Waiting for the next timepoint and position")
            position_name, time_point, timeout = self.wait_for_pause(timeout_ms=timeout_ms)
            if timeout : 
                continue
            if position_name not in self.position_names :
                self.continue_from_pause()
                continue

            if self.stop_requested :
                return

            # Read image
            PosSetting = self.pos_to_PosSettings[position_name]
            channel = self.pos_to_Channel[position_name]

            image = self.read_image(PosSetting, channel, time_point)
            return image, time_point, PosSetting
        
    def read_image(self, PosSetting, channel, time_point) :

        # Get path
        image_path = os.path.join(self.positions_config[PosSetting]["images_dir"], f"t{time_point:04}_{channel}.tif")

        if not os.path.exists(image_path):
            self.logger.error(f"Missing image at{image_path}")
            return None
        try :
            image = tifffile.imread(str(image_path))
            self.logger.info(f"Read image {image_path}")
            self.logger.info(f"Image shape : {image.shape}")
            return image
        except Exception as e:
            self.logger.error(f'Cannot read {image_path}: {e}')
            return None


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
            position_y=pos.position_y + shift_y,  # Y Axis is inverted for the microscope stage coordinates
            position_z=pos.position_z - shift_z,
        )

    def connect(self) :
        self.microscope.connect()

    def disconnect(self) :
        self.microscope.disconnect()

    def stop(self) :
        self.logger.info("Stop")
        self.stop_requested = True
        self.microscope.disconnect()


class SimulatedMicroscopeInterface_LS1 :
    def __init__(self, positions_config, max_timeout=8) :
        self.positions_config = positions_config
        # Get the position names seperated from the settings
        self.position_names = [posSetting.split("_")[0] for posSetting in self.positions_config.keys()]
        # Make the position to PositionSettings and channel lookup table
        self.pos_to_PosSettings = {}
        self.pos_to_Channel = {}
        for pos_name in positions_config.keys() :
            position_setting = pos_name.split("_")
            self.pos_to_PosSettings[position_setting[0]] = pos_name
            self.pos_to_Channel[position_setting[0]] = positions_config[pos_name]["filename"].replace(".tif","").split("_")[-1]
        
        self.nb_positions = len(self.position_names)
        self.current_position_index = 0
        self.timepoint = 0
        self.timeout_count = 1
        # Send x timeouts before pausing
        self.max_timeout = max_timeout
        # Set default logger
        self.logger = init_logger(self.__class__.__name__)
        self.stop_requested = False


    # Waits for a new image
    def wait_for_image(self, timeout_ms) :
        timeout = False
        while not self.stop_requested :
            
            if not timeout:
                self.logger.info(f"Waiting for the next timepoint and position")
            position_name, time_point, timeout = self.wait_for_pause(timeout_ms=timeout_ms)
            if timeout : 
                continue
            if position_name not in self.position_names :
                self.continue_from_pause()
                continue

            if self.stop_requested :
                return

            # Read image
            PosSetting = self.pos_to_PosSettings[position_name]
            channel = self.pos_to_Channel[position_name]

            image = self.read_image(PosSetting, channel, time_point)
            return image, time_point, PosSetting
        
    def read_image(self, PosSetting, channel, time_point) :
        # Get path
        image_path = os.path.join(self.positions_config[PosSetting]["images_dir"], f"t{time_point:04}_{channel}.tif")

        if not os.path.exists(image_path):
            self.logger.error(f"Missing image at{image_path}")
            return None
        try :
            image = tifffile.imread(str(image_path))
            self.logger.info(f"Read image {image_path}")
            self.logger.info(f"Image shape : {image.shape}")
            return image
        except Exception as e:
            self.logger.error(f'Cannot read {image_path}: {e}')
            return None


    # Simulates LS1 wait for pause function
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
        self.logger.info(f"Pausing for position [{current_pos}] at timepoint {self.timepoint}")
        return current_pos, self.timepoint, False


    def pause_after_position(self) :
        self.logger.info("Pause after position")
        return
    
    def no_pause_after_position(self) :
        self.logger.info("No pause after position")
        return
    
    def continue_from_pause(self) :
        self.logger.info("Continue from pause")
        return
    
    def relative_move(self, position_name, shift_x, shift_y, shift_z) :
        self.logger.info(f"Relative move :[{position_name}], x={shift_x}, y={shift_y}, z={shift_z}")
        return
    
    def connect(self) :
        self.logger.info("Connect")
        return
    
    def disconnect(self) :
        self.logger.info("Disconnect")
        return
    
    def stop(self) :
        self.logger.info("Stop")
        self.stop_requested = True

    
class SimulatedMicroscopeInterface_General :
    def __init__(self, positions_config) :
        self.positions_config = positions_config
        self.position_names = list(self.positions_config.keys())
        self.nb_positions = len(self.position_names)
        self.logger = init_logger(self.__class__.__name__)
        # Get the naming format
        self.nb_digits = self.detect_format(self.positions_config[next(iter(self.positions_config))]["filename"])
        self.timepoint = 0
        self.current_position_index = 0

    def wait_for_image(self, timeout_ms=100) :
        time.sleep(timeout_ms/1000)
        position_name, timepoint = self.get_pos_timepoint()
        image = self.read_image(position_name, timepoint)
        return image, timepoint, position_name


    def read_image(self, position_name, timepoint) :
        nb_zeros = self.nb_digits - len(str(timepoint))
        filename = "t" + "0" * nb_zeros + str(timepoint) + ".tif"
        image_dir = self.positions_config[position_name]["images_dir"]
        image_path = os.path.join(image_dir, filename)

        if not os.path.exists(image_path):
            self.logger.error(f"Missing image at{image_path}")
            return None
        try :
            image = tifffile.imread(str(image_path))
            self.logger.info(f"Read image {image_path}")
            self.logger.info(f"Image shape : {image.shape}")
            return image
        except Exception as e:
            self.logger.error(f'Cannot read {image_path}: {e}')
            return None

    def get_pos_timepoint(self) :
        # Go through positions in a round robin cycle
        current_pos = self.position_names[self.current_position_index]
        # Update timepoint if after a full cycle
        if self.current_position_index == 0 :
            self.timepoint = self.timepoint + 1
        # Update position for the next call
        self.current_position_index = (self.current_position_index + 1) % self.nb_positions
        self.logger.info(f"Position [{current_pos}] at timepoint {self.timepoint}")
        return current_pos, self.timepoint


    def detect_format(self, filename) :
        import re
        match = re.match(r"t(\d+)\.tif$", filename)
        if not match:
            self.logger.info(f"Could not match a filename with format t(\d+)\.tif$, {filename}")
        digits = match.group(1)
        return len(digits)
    
    def relative_move(self, position_name, shift_x, shift_y, shift_z) :
        self.logger.info(f"Relative move :[{position_name}], x={shift_x}, y={shift_y}, z={shift_z}")
        return
        