# Tracking cotracker
import os
import sys
import json
import logging
import yaml
LAUNCH_DIR = os.getcwd()

# executed if script started from terminal
if __name__ == "__main__":
    PYMCSDIR=sys.argv[1]
    python_interpreter = sys.executable
    print('PYMCS directory    : ',PYMCSDIR)
    print('python interpreter : ',python_interpreter)
    if LAUNCH_DIR not in sys.path:
        sys.path.insert(0, LAUNCH_DIR)
    from interactive_tools import bokeh_selection
    import threading
    thread = threading.Thread(target=bokeh_selection.run_server)
    thread.start()
    print("Bokeh is now serving at http://localhost:5020/")
    #script_gui = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "gui", "script_gui.py"))
    script_gui = os.path.join(PYMCSDIR, "gui", "script_gui.py")
    print('script_gui  ',script_gui)
    script_name = os.path.splitext(os.path.basename(os.path.abspath(__file__)))[0]
    os.spawnl(os.P_WAIT, python_interpreter, python_interpreter, script_gui, script_name)
    sys.exit()


# Script with user interface must implement a Script class.
# Script class must include properties defining parameters
# shown in user interface and run and stop methods.
class Script:
    def __init__(self):
        # list of unique parameter identifiers
        self.parameters = [
            "pixel_size_xy", "pixel_size_z", "dirpath", "serverkit", "scaling_factor", "simulated_microscope"
        ]

        # label of each parameter in user interface
        self.parameter_label = {
            "pixel_size_xy" : "Pixel size x, y",
            "pixel_size_z" : "Step size z",
            "dirpath": "Data directory",
            "serverkit" : "Use serverkit",
            "simulated_microscope" : "Simulated Microscope",
            "scaling_factor": "Scaling Factor",
        }

        # default parameters values loaded to user interface
        self.parameter_default_value = {
            "pixel_size_xy" : 0.347,
            "pixel_size_z" : 1.0,
            "dirpath": "",
            "serverkit": True,
            "simulated_microscope": False,
            "scaling_factor" : 2,
        }
        
        # types of parameters, min and max values provided in tuple
        self.parameter_type = {
            "pixel_size_xy" : float,
            "pixel_size_z" : float,
            "dirpath": str,
            "serverkit": bool,
            "simulated_microscope": bool,
            "scaling_factor" : int,
        }

        # control to use in user interface, if not defined text box will be used
        self.parameter_control = {
            "dirpath": "folder",
            "log_dir": "folder",
            "serverkit": "checkbox",
            "simulated_microscope": "checkbox",
            "scaling_factor": ("combobox", [1, 2, 3, 4], "normal"),
        }
        
        # documentation shown in user interface for each parameter
        self.parameter_help = {
            "pixel_size_xy": "Pixel size in x and y",
            "pixel_size_z": "Step size in z",
            "dirpath": "Timelapse folder",
            "serverkit": "Use the serverkit implementation for the models",
            "simulated_microscope": "Use a simulated microscope for offline debugging",
            "scaling_factor" : "Number of downscaling applied to the images"
        }

        self.window_tittle = "Tracking test"
        
        # indicates that the run method finnished before handling mouse wheel event
        self._figure_data_complete = False

        self._number_of_figures = 0
        self._figure_data = None
        self._figure_color_indexes = []
        self._available_colors = ['b', 'r', 'g', 'y']

    def run(self, parameter_values, update_figures_callback):
        """
        Called by the user interface to run the script.

        Parameters
        ----------
        parameter_values : list
            List of parameter values.
        figures : list
            List of figure(s) on which the script can plot.
        update_figure_callback : method
            Callback method to update figure in the user interface.

        Notes
        -----
        The user interface initializes empty figures and provides them
        for plotting to this method. To update figures in the user interface
        update_figure_callback should be called
        """

        if LAUNCH_DIR not in sys.path:
            sys.path.insert(0, LAUNCH_DIR)

        self._figure_data_complete = False
        self.stop_requested = False
        
        print("Script started.")
        print("Parameters:\n" + str(parameter_values))
        print("Script running:")

        pixel_size_xy = parameter_values['pixel_size_xy'] 
        pixel_size_z = parameter_values['pixel_size_z'] 
        serverkit = parameter_values['serverkit']
        simulated_microscope = parameter_values['simulated_microscope']
        scaling_factor = parameter_values['scaling_factor']

        # dirpath 
        dirpath = parameter_values['dirpath'] # Timelapse folder

        # Load config 
        config_path = os.path.join(os.path.dirname(__file__), 'tracking_tools', 'tracking_config.yaml')
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

        except FileNotFoundError:
            config_path = os.path.join(os.getcwd(), 'tracking_tools', 'tracking_config.yaml')
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

        roi_tracker_config = config['roi_tracker']
        roi_tracker_config["serverkit"] = serverkit
        roi_tracker_config["scaling_factor"] = scaling_factor
        position_tracker_config = {}
        position_tracker_config["pixel_size_xy"] = pixel_size_xy
        position_tracker_config["pixel_size_z"] = pixel_size_z
        runner_config = config['tracking_runner']
        simulation_config = config['simulated_microscope']


        from tracking_tools.tracking_runner.TrackingRunner import TrackingRunner
        from tracking_tools.microscope_interface.MicroscopeInterface import MicroscopeInterface, SimulatedMicroscopeInterface
        from tracking_tools.position_tracker.PositionTracker import PositionTrackerSingleRoI_v2
        from tracking_tools.image_reader.ImageReader import ImageReader
        log_dir_name = runner_config['log_dir_name']
        position_config = self.search_JSON_files(dirpath, log_dir_name)
        print(position_config)

        self.setup_global_logging(dirpath)

        if simulated_microscope :
            microscope = SimulatedMicroscopeInterface(position_names=[v['Position'] for v in position_config.values()], **simulation_config)
        else :
            microscope = MicroscopeInterface()

        self.microscope = microscope

        position_tracker = PositionTrackerSingleRoI_v2
        
        image_reader = ImageReader

        self.runner = TrackingRunner(
            microscope_interface=microscope,
            position_tracker=position_tracker,
            image_reader=image_reader,
            positions_config=position_config,
            dirpath=dirpath,
            runner_params=runner_config,
            roi_tracker_params=roi_tracker_config,
            position_tracker_params=position_tracker_config,
        )
        self.runner.run()

        print("")
        microscope.no_pause_after_position()
        microscope.disconnect()
        print("Script stopped.")

    def stop(self):
        """
        Called by user interface to stop script.

        Notes
        -----
        Run method should terminate once stop is called.
        """
        self.stop_requested = True
        self.runner.stop_requested = True
        self.microscope.no_pause_after_position()
        self.microscope.disconnect()
    
    
    def get_figure_count(self, parameter_values):
        """
        Called by user interface to get the number of figures to generate.

        Parameters
        ----------
        parameter_values : list
            List of parameter values.
        """
        return self._number_of_figures


    def figure_event(self, event_type, event_data, figure_idx, update_figures_callback):
        """
        Called by the user interface after an event on figure was raised.

        Parameters
        ----------
        event_type : string
            Name of the event, supported types are: 'MouseWheelUp', 'MouseWheelDown'
        event_data : string
            Data transmitted from the event, for key press event it's a character.
        figure_idx : int
            Index of figure on which event was raised.
        update_figure_callback: method
            Callback method to update figures in the user interface.
        """
        if self._figure_data_complete:
            if event_type == "MouseWheelUp":
                if self._figure_color_indexes[figure_idx] < len(self._available_colors) - 1: 
                    self._figure_color_indexes[figure_idx] += 1
                else:
                    self._figure_color_indexes[figure_idx] = len(self._available_colors) - 1
                update_figures_callback()

            if event_type == "MouseWheelDown":
                if self._figure_color_indexes[figure_idx] >= 1: 
                    self._figure_color_indexes[figure_idx] -= 1
                else:
                    self._figure_color_indexes[figure_idx] = 0
                update_figures_callback()


    def update_figures(self, figures):
        """
        Update the figure(s).

        Parameters
        ----------
        figures : list
            List of the figures to draw on.
        """
        for figure_index in range(len(figures)):
            figures[figure_index].clf()
            figures[figure_index].add_subplot(1,1,1).plot(self._figure_data[0], self._figure_data[1], self._available_colors[self._figure_color_indexes[figure_index]])
            figures[figure_index].tight_layout()  

    def search_JSON_files(self, dirpath, log_dir_name) :
        import glob
        import os
        import json
        positions_config = {}
        file_list = glob.glob(os.path.join(dirpath, '*', log_dir_name, 'tracking_RoIs.json'))
        print(file_list)
        for file in file_list :
            PosSettingsName = os.path.split(os.path.split(os.path.split(file)[0])[0])[1]
            Pos, Settings = PosSettingsName.split('_')
            print(file)
            with open(file) as json_data:
                d = json.load(json_data)
                json_data.close()
            print(d)
            channel = d['channel']
            positions_config[PosSettingsName] = {
                'Position' : Pos,
                'Settings' : Settings,
                'RoIs' : d['RoIs'],
                'use_detection': d['detection'],
                'channel': channel
            }
        return positions_config
    
    @staticmethod
    def setup_global_logging(log_dir):
        # Fixed filename
        log_filename = 'log_output.log'
        # Make sure the log_dir exists
        os.makedirs(log_dir, exist_ok=True)
        # Full path to the log file
        log_file_path = os.path.join(log_dir, log_filename)
        # Get the root logger
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        # Clear existing handlers to avoid duplicates
        if logger.hasHandlers():
            logger.handlers.clear()
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '[%(asctime)s] %(name)s %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        # File handler
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '[%(asctime)s] %(name)s %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
