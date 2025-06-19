from tracking_tools.tracking_runner.TrackingRunner import TrackingRunner
from tracking_tools.image_reader.ImageReader import ImageReader
from tracking_tools.position_tracker.PositionTracker import PositionTrackerSingleRoI_v2
from tracking_tools.microscope_interface.MicroscopeInterface import SimulatedMicroscopeInterface
from pathlib import Path

dirpath = Path("/home/pili/Desktop/automatic-tail-tracking/Tail detection/data/long_tracking_tests/20250515_141408_Experiment/")
dirpath = Path("/Users/helsens/Software/github/EPFL-TOP/lightsheet-live-tracking-tool/20250515_141408_Experiment/")
### Helper function to retrieve the initialization ROIs
def search_JSON_files(dirpath, log_dir_name) :
    import glob
    import os
    import json
    positions_config = {}
    file_list = glob.glob(os.path.join(dirpath, '*', log_dir_name, 'tracking_RoIs.json'))
    for file in file_list :
        PosSettingsName = os.path.split(os.path.split(os.path.split(file)[0])[0])[1]
        Pos, Settings = PosSettingsName.split('_')
        print(file)
        with open(file) as json_data:
            d = json.load(json_data)
            json_data.close()
        print(d)
        try :
            detection = d['detection']
        except Exception as e :
            print('No detection available, setting detection to false')
            detection = False
        channel = d['channel']
        positions_config[PosSettingsName] = {
            'Position' : Pos,
            'Settings' : Settings,
            'RoIs' : d['RoIs'],
            'use_detection': detection,
            'channel': channel
        }
    return positions_config

### Configs
runner_config = {
    "timeout_ms": 0,
    "log": True,
    "log_dir_name": "embryo_tracking",
}
position_tracker_config = {
    "pixel_size_xy": 0.347,
    "pixel_size_z": 1,
}
roi_tracker_config = {
    "window_length": 10,
    "grid_size": 40,
    "scaling_factor": 2,
    "server_addresses": ['http://upoates-tethys.epfl.ch:8000/', 'http://paperino.epfl.ch:8000'], # List of server addresses for remote GPU execution.
    "base_kernel_size_xy": 41,
    "kernel_size_z": 5,
    "containment_threshold": 0.4,
    "k": 5.0,
    "c0": 0.4,
    "size_ratio_threshold": 0.3,
    "score_threshold": 0.9,
    "model_path": "default",
    "serverkit": True,  # Choose wether to use imageing-server-kit
}

position_config = search_JSON_files(dirpath, runner_config["log_dir_name"])

# Choose which position to track with the simulated microscope
position_names = ["Position 1"]

microscope = SimulatedMicroscopeInterface(position_names, max_timeout=1)
position_tracker = PositionTrackerSingleRoI_v2
image_reader = ImageReader
runner = TrackingRunner(
    microscope_interface=microscope,
    position_tracker=position_tracker,
    image_reader=image_reader,
    positions_config=position_config,
    dirpath=dirpath,
    runner_params=runner_config,
    roi_tracker_params=roi_tracker_config,
    position_tracker_params=position_tracker_config
)
runner.run()



  


