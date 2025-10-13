import tkinter as tk
from tornado.ioloop import IOLoop
from tkinter import filedialog
from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Button, Checkbox, Div, Range1d, Slider, RangeSlider, LinearColorMapper
import base64
from PIL import Image
import io
from bokeh.layouts import column, row
from bokeh.events import SelectionGeometry
from bokeh.server.server import Server
import numpy as np
import json, os, pathlib, glob, sys
import tifffile
import socket
import panel as pn
import torch
import matplotlib.cm as cm
from skimage.draw import line
import pandas as pd
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from  training_tools import tail_detection_visu as tdv
except ModuleNotFoundError: 
    print("Module 'tail_detection_visu' not found. Ensure the training_tools package is installed or available in the PYTHONPATH.")

try:
    from  tracking_tools.utils import tracking_utils as tutils
except ModuleNotFoundError: 
    print("Module 'tail_detection_visu' not found. Ensure the training_tools package is installed or available in the PYTHONPATH.")



def make_layout():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    model = model.to(device)
    tracks = None
    tracks_overlays = []
    all_images = []
    all_points = []
    folder = None
    ####### GENERAL WIDGETS ##########
    # Sliders
    contrast_slider = RangeSlider(start=0, end=255, value=(0, 255), step=1, title="Contrast", width=150)
    timepoint_slider = Slider(start=0, end=0, value=0, step=1, title="Timepoint", width=250)
    tracks_slider = Slider(start=0, end=0, value=0, step=1, title="Timepoint", width=250)

    # Buttons
    select_folder_button = Button(label="Browse Folder...", button_type="primary")
    run_tracking_button = Button(label="Run CoTracker...", button_type="success")
    save_tracks_button = Button(label="Save tracks", button_type="primary")

    # Checkbox
    leave_trace_checkbox = Checkbox(label="Leave traces", active=True)
    display_points_checkbox = Checkbox(label="Display tracked points", active=False)

    # Texts
    status = Div(text="")
    model_status = Div(text="")
    tracks_status = Div(text="")

    ############ DATA SOURCES #############
    # Initial dummy image
    initial_img = np.random.randint(0, 255, (10, 1000, 1000), dtype=np.uint8)[::-1]

    # Displayed image data source 
    displayed_source = ColumnDataSource(data=dict(
        image=[initial_img[0]], x=[0], y=[0], dw=[initial_img.shape[1]], dh=[initial_img.shape[0]]
    ))

    tracks_displayed_source = ColumnDataSource(data=dict(
        image=[initial_img[0]], x=[0], y=[0], dw=[initial_img.shape[1]], dh=[initial_img.shape[0]]
    ))

    points_source = ColumnDataSource(data=dict(
        x=[], y=[],
    ))

    overlay_source = ColumnDataSource(data=dict(
        image=[], x=[], y=[], dw=[], dh=[]
    ))

    tracks_overlay_source = ColumnDataSource(data=dict(
        image=[], x=[], y=[], dw=[], dh=[]
    ))

    tracks_points_source = ColumnDataSource(data=dict(
        x=[], y=[]
    ))


    ########## FIGURE SETUP ############
    p = figure(
        title="Tracking points selector",
        x_range=(0, initial_img.shape[1]), y_range=(0, initial_img.shape[0]),
        tools="pan,wheel_zoom,reset,undo,redo,tap",
        match_aspect=True,
        width=800, height=800
    )

    p_tracks = figure(
        title="Tracks visualisation",
        x_range=(0, initial_img.shape[1]), y_range=(0, initial_img.shape[0]),
        match_aspect=True,
        width=800, height=800
    )

    # Display image from source
    color_mapper = LinearColorMapper(palette="Greys256", low=0, high=255)
    p.image('image', x='x', y='y', dw='dw', dh='dh', source=displayed_source,  color_mapper=color_mapper)
    p_tracks.image('image', x='x', y='y', dw='dw', dh='dh', source=tracks_displayed_source,  color_mapper=color_mapper)

    # Display the tracks points
    p_tracks.circle("x", "y", source=tracks_points_source, size=6, color="red", selection_alpha=1.0, nonselection_alpha=1.0)

    # RGBA overlay
    overlay_glyph = p.image_rgba(
        "image", "x", "y", "dw", "dh", alpha=0.5, source=overlay_source
    )

    tracks_overlay_glyph = p_tracks.image_rgba(
        "image", "x", "y", "dw", "dh", alpha=0.8, source=tracks_overlay_source
    )


    ############## CORE FUNCTIONS ##############
    #_______________________________________________________________________________________________
    def downscale_image(image, scaling_factor) :
        output = image[::scaling_factor, ::scaling_factor]
        # print(f"Downscaling image : {image.shape} -> {output.shape}")
        return output
    
    #_______________________________________________________________________________________________
    def normalize_image(image) :
        # MinMax normalization and conversion to uint8
        max_value = np.max(image)
        min_value = np.min(image)
        range = max_value - min_value
        normalized = (image - min_value) / range
        normalized = (normalized * 255).astype(np.uint8)
        return normalized
    
    #_______________________________________________________________________________________________
    def update_display(attr, old, new) :
        timepoint = timepoint_slider.value
        working = all_images[timepoint]
        # Normalize image
        displayed = normalize_image(working)
        # Flip image
        displayed = np.flip(displayed, axis=0)
        # Fill data source
        displayed_source.data = dict(
            image=[displayed], x=[0], y=[0], dw=[displayed.shape[1]], dh=[displayed.shape[0]]
        )
        # Update figure range
        x_range = Range1d(start=0, end=displayed.shape[0])
        y_range = Range1d(start=0, end=displayed.shape[1])
        p.x_range=x_range
        p.y_range=y_range

    #________________________________________________________________________________________________
    def update_tracks_display(attr, old, new) :
        timepoint = tracks_slider.value
        working = all_images[timepoint]
        # Normalize image
        displayed = normalize_image(working)
        # Flip image
        displayed = np.flip(displayed, axis=0)
        # Fill data source
        tracks_displayed_source.data = dict(
            image=[displayed], x=[0], y=[0], dw=[displayed.shape[1]], dh=[displayed.shape[0]]
        )
        # Update figure range
        x_range = Range1d(start=0, end=displayed.shape[0])
        y_range = Range1d(start=0, end=displayed.shape[1])
        p_tracks.x_range=x_range
        p_tracks.y_range=y_range


    #___________________________________________________________________________________________
    def update_contrast(attr, old, new):
        low, high = new 
        color_mapper.low = low
        color_mapper.high = high


    #_______________________________________________________

    _root = tk.Tk()
    _root.withdraw()

    def _get_parent():
        win = tk.Toplevel(_root)
        win.overrideredirect(True)
        win.geometry("1x1+200+200")
        win.lift()
        win.attributes("-topmost", True)
        win.focus_force()
        return win


    def select_folder():
        nonlocal folder
        parent = _get_parent()
        folder = filedialog.askdirectory(parent=parent)
        parent.destroy()

        if folder:
            status.text = f"{folder} selected" 
            load_images(folder)           
        else:
            status.text = "No directory selected."

    #________________________________________________________
    def list_images(folder) :
        files = os.listdir(folder)
        tif_files = [file for file in files if file.endswith(".tif")]
        tif_files.sort()
        tif_files = [os.path.join(folder, file) for file in tif_files]
        return tif_files
    
    #________________________________________________________
    def load_images(folder) :
        nonlocal all_images
        nonlocal all_points
        tif_files = list_images(folder)
        all_images = []
        for file in tif_files :
            try :
                im = tifffile.imread(file)
                if im.ndim != 2 :
                    print("Images are not 2D")
                    break
                all_images.append(im)
                print(f"Loaded image : {file}")
            except Exception as e:
                print(f"Could not read image : {e}")
        timepoint_slider.end = len(all_images) - 1
        tracks_slider.end = len(all_images) - 1

        working = all_images[0]
        # Normalize image
        displayed = normalize_image(working)
        # Flip image
        displayed = np.flip(displayed, axis=0)
        # Fill data sources
        displayed_source.data = dict(
        image=[displayed], x=[0], y=[0], dw=[displayed.shape[1]], dh=[displayed.shape[0]]
        )
        tracks_displayed_source.data = dict(
        image=[displayed], x=[0], y=[0], dw=[displayed.shape[1]], dh=[displayed.shape[0]]
        )
        # Update figures range
        x_range = Range1d(start=0, end=displayed.shape[0])
        y_range = Range1d(start=0, end=displayed.shape[1])
        p.x_range=x_range
        p.y_range=y_range
        p_tracks.x_range=x_range
        p_tracks.y_range=y_range

        all_points = [{"x":[], "y":[]} for _ in range(len(all_images))]

    #________________________________________________________
    def tap_callback(event):
        timepoint = timepoint_slider.value
        points = all_points[timepoint]
        new_x, new_y = int(event.x), int(event.y) + 1
        coords = set(zip(points["x"], points["y"]))

        if (new_x, new_y) in coords :
            coords.remove((new_x, new_y))
        else :
            coords.add((new_x, new_y))

        if coords :
            xs, ys = zip(*coords)
        else :
            xs, ys = [], []

        points_source.data = dict(x=list(xs), y=list(ys))
        all_points[timepoint] = dict(x=list(xs), y=list(ys))

    #________________________________________________________
    def update_points(attr, old, new) :
        timepoint = timepoint_slider.value
        points = all_points[timepoint]
        points_source.data = dict(
            x=points["x"], y=points["y"]
        )

        
    #________________________________________________________
    def binary2rgba(binary, color=(0, 255, 255), alpha=255):
        # Alpha is set to maximum here and changed via the glyph object
        h, w = binary.shape
        r, g, b = color
        rgba = np.zeros((h, w), dtype=np.uint32)

        # Fill RGBA Mask :  alpha     red     green    blue
        #                  xxxxxxxx xxxxxxxx xxxxxxxx xxxxxxxx 
        rgba[binary.astype(bool)] = (alpha << 24) | (r << 16) | (g << 8) | b
        return np.flip(rgba, axis=0)
    
    #________________________________________________________________________________________________
    def update_overlay(attr, old, new) :
        nonlocal all_images
        # Create binary mask
        H, W = all_images[0].shape
        overlay = np.zeros((H, W))
        x_coords = points_source.data["x"]
        y_coords = points_source.data["y"]
        for x, y in zip(x_coords, y_coords) :
            overlay[H - y][x] = 1
        # Convert to rgba and update overlay
        overlay_rgba = binary2rgba(overlay)
        overlay_source.data = dict(
            image=[overlay_rgba], x=[0], y=[0], dw=[overlay_rgba.shape[1]], dh=[overlay_rgba.shape[0]]
        )

    #_________________________________________________________________
    def run_tracking_callback() :
        nonlocal tracks
        nonlocal tracks_overlays
        nonlocal all_images
        # model_status.text = "Computing tracks..."
        H, W = all_images[0].shape

        # Build queries and video
        queries = []
        for timepoint, coords in enumerate(all_points) :
            for x, y in zip(coords["x"], coords["y"]) :
                queries.append([timepoint, x, H-y])
        queries = np.array(queries)
        queries_tensor = torch.tensor(queries, dtype=torch.float16, device=device)[None]
        queries_tensor = queries_tensor.to(device)

        video = np.array(all_images)
        video_chunk = _prepare_video_chunk(video)
        video_chunk = video_chunk.to(device)

        # Run model
        pred_tracks, _ = model(video_chunk, queries=queries_tensor, backward_tracking=True)
        pred_tracks_np = pred_tracks.cpu().numpy()[0]
        tracks = pred_tracks_np

        # Update tracks overlays
        # For each timepoint, draw the tracks until that timepoint
        N = tracks.shape[0]
    
        tracks_overlays = []
        for t in range(N) :
            overlay = tracks2rgba(tracks, t, H, W)
            tracks_overlays.append(overlay)

        tracks_slider.value = 0
        tracks_overlay_source.data = dict(
            image=[tracks_overlays[0]]
        )
        # model_status.text="Finished computing tracks"
        run_tracking_button.label = "Run CoTracker..."
        run_tracking_button.button_type = "success"


    #___________________________________________________________________________________________
    def run_tracking_callback_short():
        run_tracking_button.label = "Processing..."
        run_tracking_button.button_type = "danger"
        curdoc().add_next_tick_callback(run_tracking_callback)


    #_______________________________________________________________________
    def update_tracks_overlay(attr, old, new) :
        nonlocal tracks_overlays
        nonlocal tracks
        nonlocal all_images
        H, W = all_images[0].shape

        if display_points_checkbox.active :
            tracks_points = tracks[tracks_slider.value]
            tracks_points_source.data = dict(
                x=list(tracks_points[:,0]), y=list(H-tracks_points[:,1])
            )
        if tracks_overlays and leave_trace_checkbox.active :
            overlay_rgba = tracks_overlays[tracks_slider.value]
            tracks_overlay_source.data = dict(
                image=[overlay_rgba], x=[0], y=[0], dw=[overlay_rgba.shape[1]], dh=[overlay_rgba.shape[0]]
            )


    #_______________________________________________________________________
    def tracks2rgba(tracks, timepoint, H, W, colors=None, alpha=255) :
        n_tracks = tracks.shape[1]
        if colors is None:
            cmap = cm.tab10(np.linspace(0, 1, n_tracks))
            colors = [(int(r*255), int(g*255), int(b*255)) for r, g, b, _ in cmap]

        rgba = np.zeros((H, W), dtype=np.uint32)

        for tracks_idx in range(n_tracks) :
            pts = tracks[:timepoint+1, tracks_idx, :]

            if len(pts) > 1:
                r, g, b = colors[tracks_idx]
                argb_val = (alpha << 24) | (r << 16) | (g << 8) | b

                for i in range(len(pts)-1):
                    x1, y1 = pts[i]
                    x2, y2 = pts[i+1]
                    rr, cc, = line(int(y1), int(x1), int(y2), int(x2))

                    mask = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
                    rr, cc = rr[mask], cc[mask]

                    rgba[rr, cc] = argb_val
        
        return np.flip(rgba, axis=0)

    #_____________________________________________________________________
    def _prepare_video_chunk(window_frames) :
        frames = np.asarray(window_frames.copy())
        if frames.ndim == 3 : # If not RGB (no channel dimension)
            frames = np.repeat(frames[..., np.newaxis], 3, axis=-1) # Convert to RGB by duplicating into 3 channels
        video_chunk = torch.tensor(
            np.stack(frames), device=device
            ).float().permute(0, 3, 1, 2)[None]
        return video_chunk
    
    #_______________________________________________________________________
    def leave_traces_callback(attr, old, new) :
        nonlocal tracks_overlays
        if tracks_overlays and leave_trace_checkbox.active==True :
            overlay_rgba = tracks_overlays[tracks_slider.value]
            tracks_overlay_source.data = dict(
                image=[overlay_rgba], x=[0], y=[0], dw=[overlay_rgba.shape[1]], dh=[overlay_rgba.shape[0]]
            )
        else :
            tracks_overlay_source.data = dict(
                image=[], x=[], y=[], dw=[], dh=[]
            ) 
    
    #__________________________________________________________________________
    def display_points_callback(attr, old, new) :
        nonlocal tracks
        nonlocal all_images
        H, W = all_images[0].shape
        if display_points_checkbox.active :
            tracks_points = tracks[tracks_slider.value]
            tracks_points_source.data = dict(
                x=list(tracks_points[:,0]), y=list(H-tracks_points[:,1])
            )
        else :
            tracks_points = tracks[tracks_slider.value]
            tracks_points_source.data = dict(
                x=[], y=[]
            )

    #_____________________________________________________________________________
    def save_tracks_callback() :
        nonlocal tracks
        n_timepoints, n_tracks, _ = tracks.shape
        rows = []
        for t in range(n_timepoints) :
            for tracks_id in range(n_tracks) :
                x, y = tracks[t, tracks_id]
                rows.append({
                    "timepoint": t,
                    "tracks_id": tracks_id,
                    "x": x,
                    "y": y
                })

        df = pd.DataFrame(rows)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(folder, f"tracks_{timestamp}.csv")
        df = df.sort_values(by=["timepoint", "tracks_id"])
        df.to_csv(path, index=False)
        tracks_status.text = f"Tracks saved in [{path}]"

    ########## CALLBACKS #########

    contrast_slider.on_change('value', update_contrast)
    select_folder_button.on_click(select_folder)
    timepoint_slider.on_change("value", update_display, update_points)
    tracks_slider.on_change("value", update_tracks_display, update_tracks_overlay)
    p.on_event("tap", tap_callback) 
    points_source.on_change("data", update_overlay)
    run_tracking_button.on_click(run_tracking_callback)
    leave_trace_checkbox.on_change("active", leave_traces_callback)
    display_points_checkbox.on_change("active", display_points_callback)
    run_tracking_button.on_click(run_tracking_callback_short)
    save_tracks_button.on_click(save_tracks_callback)


    ########## LAYOUT ##########
    #___________________________________________________________________________________________
    def mk_div(**kwargs):
        return Div(text='<div style="background-color: white; width: 20px; height: 1px;"></div>', **kwargs)
        
    timepoint_slider_layout = row(mk_div(),timepoint_slider)
    tracks_slider_layout = row(mk_div(),tracks_slider)
    status_layout = row(mk_div(), status)

    selection_layout = column(p, timepoint_slider_layout, status_layout)
    vis_layout = column(p_tracks, tracks_slider_layout, tracks_status)
    commands_layout = column(contrast_slider, run_tracking_button, model_status, leave_trace_checkbox, display_points_checkbox, save_tracks_button)

    layout = row(
        mk_div(),
        column(
            mk_div(), 
            select_folder_button,
            row(
                selection_layout, commands_layout, vis_layout
            )
        )
    )
    
    return layout




#_______________________________________________________
def make_document(doc):
    layout = make_layout()
    doc.title = 'Tracking selection'
    doc.add_root(layout)




#_______________________________________________________
def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

#_______________________________________________________
def run_server():
    port = get_free_port()
    port = 5021
    print(f"Using dynamic port: {port}")
    io_loop = IOLoop.current()
    server = Server({'/': make_document},
                    io_loop=io_loop,
                    allow_websocket_origin=[f"localhost:{port}"],
                    port=port)
    server.start()
    print(f"Bokeh server running at http://localhost:{port}")
    io_loop.start()

#_______________________________________________________
if __name__ == '__main__':
    run_server()
    #to run a detached server
    #import threading
    #thread = threading.Thread(target=run_server)
    #thread.start()

pn.extension()
pane = pn.panel(make_layout)
pane.servable()