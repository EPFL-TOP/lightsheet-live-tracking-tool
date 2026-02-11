import numpy as np
import matplotlib.pyplot as plt
from tracking_tools.tracker.BaseTracker import MultiRoIBaseTracker
from tracking_tools.position_tracker.PositionTracker import PositionTrackerMultiROI
import os 
import tifffile
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

def shift3d_zero(arr, shift):

    dz, dy, dx = shift
    z, y, x = arr.shape

    out = np.zeros_like(arr)

    # Compute source/target slicing ranges
    z0 = max(0, dz);   z1 = min(z, z + dz)
    y0 = max(0, dy);   y1 = min(y, y + dy)
    x0 = max(0, dx);   x1 = min(x, x + dx)

    out[z0:z1, y0:y1, x0:x1] = arr[z0-dz:z1-dz, y0-dy:y1-dy, x0-dx:x1-dx]

    return out


def shifts_list_to_array(shifts_list) :
    list_of_lists = []
    for shift in shifts_list :
        list_of_lists.append([shift.z, shift.y, shift.x])
    return np.array(list_of_lists)


def save_plots(residuals, run_dir) :
    axes = ["Z", "Y", "X"]

    # In pixels
    for i, axis_name in enumerate(axes):
        data = residuals[:, i]
        mean = np.mean(data)
        std = np.std(data)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data, kde=True, ax=ax)

        textstr = (
            f"Mean = {mean:.2f}\n"
            f"Std = {std:.2f}"
        )
        ax.text(
            0.97, 0.97, textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

        ax.set_title(f"Residuals distribution – {axis_name} axis")
        ax.set_xlabel("Residual shifts (pixels)")
        ax.set_ylabel("Counts")

        plt.tight_layout()

        filename = os.path.join(run_dir, f"residuals_{axis_name}_px.png")
        fig.savefig(filename, dpi=300, bbox_inches="tight")


        # Add lines and save again
        ax.axvline(mean, linestyle="--", linewidth=2, label=f"Mean = {mean:.2f}")
        ax.axvline(mean + std, linestyle=":", linewidth=2, label=f"+1σ = {std:.2f}")
        ax.axvline(mean - std, linestyle=":", linewidth=2, label=f"-1σ = {std:.2f}")
        ax.legend()
        plt.tight_layout()
        filename = os.path.join(run_dir, f"residuals_{axis_name}_px_with_lines.png")
        fig.savefig(filename, dpi=300, bbox_inches="tight")

        plt.close(fig)

    # In um
    residuals[:, 1:] = residuals[:, 1:] * 0.347 
    for i, axis_name in enumerate(axes):
        data = residuals[:, i]
        mean = np.mean(data)
        std = np.std(data)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data, kde=True, ax=ax)

        textstr = (
            f"Mean = {mean:.2f}\n"
            f"Std = {std:.2f}"
        )
        ax.text(
            0.97, 0.97, textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

        ax.set_title(f"Residuals distribution – {axis_name} axis")
        ax.set_xlabel("Residual shifts (µm)")
        ax.set_ylabel("Counts")

        plt.tight_layout()

        filename = os.path.join(run_dir, f"residuals_{axis_name}_um.png")
        fig.savefig(filename, dpi=300, bbox_inches="tight")


        # Add lines and save again
        ax.axvline(mean, linestyle="--", linewidth=2, label=f"Mean = {mean:.2f}")
        ax.axvline(mean + std, linestyle=":", linewidth=2, label=f"+1σ = {std:.2f}")
        ax.axvline(mean - std, linestyle=":", linewidth=2, label=f"-1σ = {std:.2f}")
        ax.legend()
        plt.tight_layout()
        filename = os.path.join(run_dir, f"residuals_{axis_name}_um_with_lines.png")
        fig.savefig(filename, dpi=300, bbox_inches="tight")

        plt.close(fig)


def run_measurement(
        position_tracker_config, 
        roi_tracker_config, 
        max_shift_xy,
        max_iterations,
        img_list, 
        roi
) :
    # Original tracker
    postracker1 = PositionTrackerMultiROI(
        rois=[roi], 
        first_frame=None, 
        log=True, 
        use_detection=False,
        position_name="Original",
        roi_tracker_params=roi_tracker_config,
        position_tracker_params=position_tracker_config,
    )

    # Tracker with shifted images
    postracker2 = PositionTrackerMultiROI(
        rois=[roi], 
        first_frame=None, 
        log=True, 
        use_detection=False,
        position_name="Shifted",
        roi_tracker_params=roi_tracker_config,
        position_tracker_params=position_tracker_config,
    )

    window_length = roi_tracker_config["window_length"]
    applied_shifts = []
    tracked_shifts = []

    # Initialize the first images
    for i in range(window_length - 1) :
        
        im = img_list[i]

        _ = postracker1.compute_shift_um(im)
        _ = postracker2.compute_shift_um(im)

    # save states
    state_window_frames = postracker1.base_tracker.window_frames
    state_rois_list = postracker1.base_tracker.rois_list
    state_tracked_points = postracker1.base_tracker.tracked_points
    state_tracks = postracker1.base_tracker.tracks
    state_count = postracker1.base_tracker.count

    # Track the last image
    im = img_list[window_length]

    _ = postracker1.compute_shift_um(im)

    # Save the shift at this state
    original_shift = postracker1.shifts_px[-1]

    # Compute shift for shifted image n times
    for i in range(max_iterations) :

        # Replace states
        postracker2.base_tracker.window_frames = state_window_frames.copy()
        postracker2.base_tracker.rois_list = state_rois_list.copy()
        postracker2.base_tracker.tracked_points = state_tracked_points.copy()
        postracker2.base_tracker.tracks = state_tracks.copy()
        postracker2.base_tracker.count = state_count

        # random translation (0 for z)
        if max_shift_xy == 0 :
            vector = np.array([0, 0, 0])
        else :
            vector = np.random.randint(low=-max_shift_xy, high=max_shift_xy, size=3)
        vector[0] = 0

        shifted = shift3d_zero(im, shift=vector)

        _ = postracker2.compute_shift_um(shifted)
        
        applied_shifts.append(vector)
        tracked_shifts.append(postracker2.shifts_px[-1])
    
    # Compute residuals
    original_shift_array = np.array([original_shift.z, original_shift.y, original_shift.x])
    tracked_shifts_array = shifts_list_to_array(tracked_shifts)
    applied_shifts_array = np.array(applied_shifts)
    residuals = original_shift_array - (tracked_shifts_array - applied_shifts_array)

    return residuals

def main(measurements_config) :
    nb_runs = measurements_config["nb_runs"]
    save_dir = measurements_config["save_dir"]
    initial_rois = measurements_config["initial_rois"]
    data_dir = measurements_config["data_dir"] 
    roi_tracker_config = measurements_config["roi_tracker_config"]
    window_length = roi_tracker_config["window_length"]
    max_shift = measurements_config["max_shift"]
    position_tracker_config = measurements_config["position_tracker_config"]
    max_iterations = measurements_config["max_iterations"]

    files = sorted(os.listdir(data_dir))
    files = [name for name in files if name.endswith(".tif")]

    assert nb_runs <= len(initial_rois)
    assert len(files) >= window_length + nb_runs

    residuals = []
    for i in range(nb_runs) :
        img_list = []

        # read imgs
        for filename in files[i:i+window_length+1] :
            im = tifffile.imread(os.path.join(data_dir, filename))
            assert(im.ndim == 2 or im.ndim == 3)
            if im.ndim == 2 :
                im = im[np.newaxis, ...]
            im = np.pad(im, pad_width=((0,0), (max_shift,max_shift), (max_shift,max_shift)), mode="constant")
            img_list.append(im)

        roi = initial_rois[i]
        roi["x"] = roi["x"] + max_shift
        roi["y"] = roi["y"] + max_shift

        residuals_run = run_measurement(
            position_tracker_config=position_tracker_config,
            roi_tracker_config=roi_tracker_config,
            max_shift_xy=max_shift,
            max_iterations=max_iterations,
            img_list=img_list,
            roi=roi
        )

        residuals.append(residuals_run)

    residuals = np.concatenate(residuals, axis=0)
    means = np.mean(residuals, axis=0)
    stds = np.std(residuals, axis=0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(save_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    np.save(os.path.join(run_dir, "residuals.npy"), residuals)
    np.save(os.path.join(run_dir, "means.npy"), means)
    np.save(os.path.join(run_dir, "stds.npy"), stds)

    print("Means", means)
    print("Stds", stds)

    save_plots(residuals, run_dir)

import argparse
import json

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to the measurements_config JSON file"
    )
    args = parser.parse_args()
    with open(args.config, "r") as f:
        measurements_config = json.load(f)
        f.close()

    main(measurements_config)

    