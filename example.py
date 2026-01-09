from tracking_tools.tracking_runner.TrackingRunner import TrackingRunner
from tracking_tools.microscope_interface.MicroscopeInterface import SimulatedMicroscopeInterface_General
from tracking_tools.utils.tracking_utils import get_pos_config
from pathlib import Path
import os
import logging


### Configs
runner_config = {
    "timeout_ms": 100,
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
    "server_addresses": ..., # List of server addresses for remote GPU execution. (imaging-server-kit)
    "base_kernel_size_xy": 41,
    "kernel_size_z": 5,
    "containment_threshold": 0.4,
    "k": 5.0,
    "c0": 0.4,
    "size_ratio_threshold": 0.3,
    "score_threshold": 0.9,
    "model_path": "default",
    "serverkit": False,  # Choose wether to use imaging-server-kit
}


def setup_global_logging(log_dir):
    log_filename = 'log_output.log'
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, log_filename)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
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



if __name__ == "__main__":

    import sys
    if len(sys.argv) < 2:
        print("Usage: python example.py <path_to_experiments_folder>")
        sys.exit(1)
    dirpath = sys.argv[1] # PATH TO THE EXERIMENTS FOLDER
    if not os.path.isabs(dirpath):
        dirpath = os.path.abspath(dirpath)
    if not os.path.exists(dirpath):
        print(f"Directory {dirpath} does not exist.")
        sys.exit(1) 

    setup_global_logging(dirpath)

    position_config = get_pos_config(dirpath, "embryo_tracking")
    print(position_config)

    microscope = SimulatedMicroscopeInterface_General(position_config)
    runner = TrackingRunner(
        microscope_interface=microscope,
        positions_config=position_config,
        dirpath=dirpath,
        runner_params=runner_config,
        roi_tracker_params=roi_tracker_config,
        position_tracker_params=position_tracker_config
    )

    runner.run_general()