import tkinter as tk
from tornado.ioloop import IOLoop
from tkinter import filedialog
from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, BoxEditTool, TapTool, LabelSet, Button, CheckboxGroup, TextInput, Div, Range1d, Slider, Select, RangeSlider, LinearColorMapper, Span
from bokeh.layouts import column, row
from bokeh.events import SelectionGeometry
from bokeh.server.server import Server
from bokeh.transform import linear_cmap
from bokeh.palettes import Viridis256
import numpy as np
import json, os, pathlib
import tifffile
import glob, sys, pickle
from PIL import Image, ImageDraw
import imageio

last_tp = -1


def make_layout():
    data_log = None
    mp_list = []

    initial_img = np.random.randint(0, 255, (1000, 1000), dtype=np.uint8)[::-1]

    image_source = ColumnDataSource(data=dict(image=[initial_img], x=[0], y=[0], dw=[initial_img.shape[1]], dh=[initial_img.shape[0]]))
    images_source = ColumnDataSource(data=dict(image=[initial_img], x=[0], y=[0], dw=[initial_img.shape[1]], dh=[initial_img.shape[0]]))
    
    shift_mu_source = ColumnDataSource(data=dict(z=[], y=[], x=[], t=[]))

    trajectory_source = ColumnDataSource(data=dict(z=[], y=[], x=[], t=[]))
    trajectory_highlight_source = ColumnDataSource(data=dict(x=[], y=[], z=[]))
    
    rects_source = ColumnDataSource(data=dict(x=[], y=[], width=[], height=[], index=[], label_x=[], label_y=[], time_point=[], tracks_idx=[]))
    rect_source  = ColumnDataSource(data=dict(x=[], y=[], width=[], height=[], index=[], label_x=[], label_y=[]))

    points_source = ColumnDataSource(data=dict(x=[], y=[]))
    point_source  = ColumnDataSource(data=dict(x=[], y=[]))

    positions=[]
    dropdown_position = Select(value="", title='Position', options=positions)

    invert_axis_checkboxgroup = CheckboxGroup(labels=["Invert x", "Invert y", "Invert z"], active=[])

    # Figure setup
    p_img = figure(
        title="RoIs tracking follower",
        x_range=(0, initial_img.shape[1]), y_range=(0, initial_img.shape[0]),
        tools="pan,wheel_zoom,box_select,reset,undo,redo",
        match_aspect=True, 
        width=800, height=800
    )

    # Figure setup
    p_shifts = figure(
        title="shifts um",
        #x_range=(0, initial_img.shape[1]), y_range=(0, initial_img.shape[0]),
        tools="pan,wheel_zoom,box_select,reset,undo,redo",
        match_aspect=True, 
        width=800, height=400,
    )

    # Trajectory Plots
    p_trajectory_xy = figure(
        title="Trajectory XY", 
        tools="pan,wheel_zoom,box_select,reset,undo,redo",
        match_aspect=True, 
        width=400, 
        height=400,
        x_axis_label="x",
        y_axis_label="y"
    )


    p_trajectory_xz = figure(
        title="Trajectory XZ", 
        tools="pan,wheel_zoom,box_select,reset,undo,redo",
        match_aspect=True, 
        width=400, 
        height=400,
        x_axis_label="x",
        y_axis_label="z"
    )
    p_trajectory_yz = figure(
        title="Trajectory YZ", 
        tools="pan,wheel_zoom,box_select,reset,undo,redo",
        match_aspect=True, 
        width=400, 
        height=400,
        x_axis_label="y",
        y_axis_label="z"
    )
    color_mapper_trajectory = linear_cmap(field_name='t', palette=Viridis256, low=0, high=1)

    # Display image from source with a color mapper for contrast
    color_mapper = LinearColorMapper(palette="Greys256", low=0, high=255)
    p_img.image('image', x='x', y='y', dw='dw', dh='dh', source=image_source,  color_mapper=color_mapper)
    p_img.scatter(x='x', y='y', source=point_source, size=5, color='red')

    slider = Slider(start=0, end=0, value=0, step=1, title="time frame", width=700)

    status = Div(text="")

    rect_glyph = p_img.rect(
        'x', 'y', 'width', 'height', source=rect_source,
        fill_alpha=0.2, fill_color='blue', line_color='red', line_width=2
    )

    # Shift values
    p_shifts.line('t','x',line_width=2, source=shift_mu_source, legend_label='x', line_color='red')
    p_shifts.line('t','y',line_width=2, source=shift_mu_source, legend_label='y', line_color='blue')
    p_shifts.line('t','z',line_width=2, source=shift_mu_source, legend_label='z', line_color='green')
    # Vertical bar
    vline = Span(location=slider.value, dimension="height", line_color="black", line_width=2, line_dash="dashed")
    p_shifts.add_layout(vline)
    # Shifts tectz display
    shifts_text = Div(text="")

    # Trajectory value
    p_trajectory_xy.scatter('x', 'y', size=8, color=color_mapper_trajectory, source=trajectory_source)
    p_trajectory_xz.scatter('x', 'z', size=8, color=color_mapper_trajectory, source=trajectory_source)
    p_trajectory_yz.scatter('y', 'z', size=8, color=color_mapper_trajectory, source=trajectory_source)
    # Trajectory highlights
    p_trajectory_xy.circle("x", "y", size=10, line_color="red", fill_color="red", line_width=2, source=trajectory_highlight_source)
    p_trajectory_yz.circle("y", "z", size=10, line_color="red", fill_color="red", line_width=2, source=trajectory_highlight_source)
    p_trajectory_xz.circle("x", "z", size=10, line_color="red", fill_color="red", line_width=2, source=trajectory_highlight_source)
    


    def mk_div(**kwargs):
        return Div(text='<div style="background-color: white; width: 20px; height: 1px;"></div>', **kwargs)


    #___________________________________________________________________________________________
    def callback_slider(attr, old, new):
        time_point = slider.value

        # Update image
        image_source.data = {'image':[images_source.data['image'][time_point]], 
                            'x':[images_source.data['x'][time_point]],
                            'y':[images_source.data['y'][time_point]],
                            'dw':[images_source.data['dw'][time_point]],
                            'dh':[images_source.data['dh'][time_point]]}
        
        # Update ROI
        if time_point>-1:
            # rect_source.data = dict(
            # x=rects_source.data['x'][time_point], 
            # y=rects_source.data['y'][time_point], 
            # width=rects_source.data['width'][time_point], 
            # height=rects_source.data['height'][time_point],
            # index=[0], label_x=[0], label_y=[0]
            # )

            rect_source.data = dict(
                x=[rects_source.data['x'][time_point][i] for i in range(len(rects_source.data['x'][time_point]))], 
                y=[rects_source.data['y'][time_point][i] for i in range(len(rects_source.data['y'][time_point]))], 
                width=rects_source.data['width'][time_point], 
                height=rects_source.data['height'][time_point], 
                index=rects_source.data['index'][time_point], 
                label_x=rects_source.data['label_x'][time_point], 
                label_y=rects_source.data['label_y'][time_point]
            )

        # Update query point
        tracks_idx = rects_source.data["tracks_idx"][time_point]
        point_source.data = dict(
            x=points_source.data['x'][tracks_idx],
            y=points_source.data['y'][tracks_idx]
        )

        # Update shifts vertical bar
        vline.location = time_point

        # Update displyed shift text
        shifts_text.text = (
            f"x (&micro;m): {shift_mu_source.data[\"x\"][slider.value]:.2f}<br>"
            f"y (&micro;m): {shift_mu_source.data[\"y\"][slider.value]:.2f}<br>"
            f"z (&micro;m): {shift_mu_source.data[\"z\"][slider.value]:.2f}"
        )

        # Update trajectory point highlight
        trajectory_highlight_source.data = {
            "x": [trajectory_source.data["x"][time_point]],
            "y": [trajectory_source.data["y"][time_point]],
            "z": [trajectory_source.data["z"][time_point]],
        }

    slider.on_change('value', callback_slider)



    #_______________________________________________________
    def downscale_image(image, n=4, order=0, verbose=False) :
        from scipy.ndimage import zoom
        initial_shape = image.shape
        for _ in range(n) :
            if image.ndim==3:
                image = zoom(image, (1, 1/2, 1/2), order=order)
            else:
                image = zoom(image, (1/2, 1/2), order=order)

        final_shape = image.shape
        if verbose :
            print(f'Lowered resolution from {initial_shape} to {final_shape}')
        return image


    #_______________________________________________________
    def create_images(dir_path, img_list):
        for img in img_list:
            print("processing image ",img)
            im =  tifffile.imread(img)
            arr = np.array(im)
            arr_down = downscale_image(arr, n=2)
            max_proj = np.max(arr_down, axis=0)
            out_name = os.path.join(dir_path,"embryo_tracking","max_proj")

            out_name = os.path.join(out_name,os.path.split(img)[1].replace('.tif', '_maxproj.tif'))
            print("-------",out_name)
            tifffile.imwrite(out_name, max_proj)


    #_______________________________________________________
    def load_images(dir_path, reload=False):
        out_name = os.path.join(dir_path,"embryo_tracking","max_proj")
        if not os.path.isdir(out_name):
            raise IsADirectoryError(f"{out_name} is not a directory")
        #     os.mkdir(out_name)
        #     img_list=glob.glob(os.path.join(dir_path,"*.tif"))
        #     create_images(dir_path, img_list)

        img_list=glob.glob(os.path.join(out_name,"*.tif"))
        img_list = sorted(img_list)
        if len(img_list) == 0 :
            raise ValueError("img_list is empty")
        print('nb images max proj: ',len(img_list))
        images_ds=[]
        x_ds=[]
        y_ds=[]
        dw_ds=[]
        dh_ds=[]

        global last_tp
        if not reload:
            last_tp = len(img_list)-1
        else:
            images_ds = images_source.data['image']
            x_ds      = images_source.data['x']
            y_ds      = images_source.data['y']
            dw_ds     = images_source.data['dw']
            dh_ds     = images_source.data['dh']

        for idx, img in enumerate(img_list):
            if reload and idx<=last_tp:
                continue
            im =  tifffile.imread(img)

            max_value = np.max(im)
            min_value = np.min(im)
            max_proj_norm = (im - min_value)/(max_value-min_value)*255
            max_proj_norm = max_proj_norm.astype(np.uint8)
            max_proj_norm = np.flip(max_proj_norm,0)

            images_ds.append(max_proj_norm)
            x_ds.append(0)
            y_ds.append(0)
            dw_ds.append(im.shape[1])
            dh_ds.append(im.shape[0])
            
            last_tp  = len(img_list)-1
        x_range = Range1d(start=0, end=images_ds[0].shape[0])
        y_range = Range1d(start=0, end=images_ds[0].shape[1])
        p_img.x_range=x_range
        p_img.y_range=y_range
        images_source.data = dict(image=images_ds, x=x_ds, y=y_ds, dw=dw_ds, dh=dh_ds)
        if not reload:
            image_source.data = dict(image=[images_ds[0]], x=[x_ds[0]], y=[y_ds[0]], dw=[dw_ds[0]], dh=[dh_ds[0]])
        




    #_______________________________________________________
    def load_tracking(dir_path, reload=False):
        nonlocal data_log
        nonlocal mp_list
        print(dir_path)
        tracking_path = os.path.join(dir_path, "embryo_tracking")
        pos_config    = os.path.join(tracking_path, "tracking_RoIs.json")
        tracking_log  = os.path.join(tracking_path, "logs.json")

        max_proj = glob.glob(os.path.join(tracking_path, "max_proj", "*max_proj.tif"))
        mp_list=[]
        for mp in max_proj:
            mp_list.append(int(os.path.split(mp)[-1].split("_")[0].replace("t","")))
        mp_list=sorted(mp_list)
        with open(pos_config, 'r') as file:
            data_pos = json.load(file)
        with open(tracking_log, 'r') as file:
            data_log = json.load(file)        

        roi_x,roi_y = [],[]
        roi_width,roi_height = [],[]
        roi_index, roi_timepoint, tracks_idx =[],[],[]
        roi_label_x,roi_label_y = [],[]
        shifts_x, shifts_y, shifts_z = [],[],[]
        scale_factor = 1

        n_tp=0
        for tp in mp_list:
            tp=str(tp)
            try:
                scale_factor = 2**data_log[tp]["scaling_factor"]
                roi_x.append([roi["x"] for roi in data_log[tp]["roi"]])
                roi_y.append([data_pos["shape"][1]/scale_factor-roi["y"] for roi in data_log[tp]["roi"]])
                roi_width.append([roi["width"] for roi in data_log[tp]["roi"]])
                roi_height.append([roi["height"] for roi in data_log[tp]["roi"]])
                roi_index.append([roi["order"] for roi in data_log[tp]["roi"]])
                roi_label_x.append([roi["x"]-roi["width"] for roi in data_log[tp]["roi"]])
                roi_label_y.append([roi["y"]+roi["height"] for roi in data_log[tp]["roi"]])
                roi_timepoint.append(tp)
                tracks_idx.append(data_log[tp]["tracks_id"])

                if 0 in invert_axis_checkboxgroup.active :
                    shifts_x.append(-data_log[tp]["shift_um"]["x"])
                else :
                    shifts_x.append(data_log[tp]["shift_um"]["x"])
                if 1 in invert_axis_checkboxgroup.active :
                    shifts_y.append(-data_log[tp]["shift_um"]["y"])
                else :
                    shifts_y.append(data_log[tp]["shift_um"]["y"])
                if 2 in invert_axis_checkboxgroup.active :
                    shifts_z.append(-data_log[tp]["shift_um"]["z"])
                else :
                    shifts_z.append(data_log[tp]["shift_um"]["z"])

                n_tp+=1
            except KeyError:
                print('no time point---',tp,'----')


        slider.end = n_tp-1
        rects_source.data = dict(x=roi_x, 
                                 y=roi_y, 
                                 width=roi_width, 
                                 height=roi_height, 
                                 index=roi_index, 
                                 label_x=roi_label_x, 
                                 label_y=roi_label_y, 
                                 time_point=roi_timepoint, 
                                 tracks_idx=tracks_idx)

        rect_source.data = dict(
            x=[rects_source.data['x'][0][i] for i in range(len(rects_source.data['x'][0]))], 
            y=[rects_source.data['y'][0][i] for i in range(len(rects_source.data['y'][0]))], 
            width=rects_source.data['width'][0], 
            height=rects_source.data['height'][0], 
            index=rects_source.data['index'][0], 
            label_x=rects_source.data['label_x'][0], 
            label_y=rects_source.data['label_y'][0]
        )
        time_axis = [i for i in range(0,len(shifts_x))]
        p_shifts.x_range = Range1d(min(time_axis), max(time_axis))

        shifts_um_cumsum_x = np.cumsum(np.array(shifts_x),axis=0)
        shifts_um_cumsum_y = np.cumsum(np.array(shifts_y),axis=0)
        shifts_um_cumsum_z = np.cumsum(np.array(shifts_z),axis=0)
        shift_mu_source.data = dict(x=shifts_x, y=shifts_y, z=shifts_z, t=[i for i in range(0,len(shifts_x))])
        shifts_text.text = (
            f"x (&micro;m): {shift_mu_source.data["x"][slider.value]:.2f}<br>"
            f"y (&micro;m): {shift_mu_source.data["y"][slider.value]:.2f}<br>"
            f"z (&micro;m): {shift_mu_source.data["z"][slider.value]:.2f}"
        )
        trajectory_source.data = dict(x=shifts_um_cumsum_x, y=shifts_um_cumsum_y, z=shifts_um_cumsum_z, t=np.arange(len(shifts_um_cumsum_x)))
        trajectory_highlight_source.data = {
            "x": [trajectory_source.data["x"][slider.value]],
            "y": [trajectory_source.data["y"][slider.value]],
            "z": [trajectory_source.data["z"][slider.value]],
        }
        n_points = len(shifts_um_cumsum_x)
        color_mapper_trajectory['transform'].low = 0
        color_mapper_trajectory['transform'].high = n_points - 1

        tracks_path = os.path.join(tracking_path,  "tracks.pkl")
        with open(tracks_path, 'rb') as f :
            tracks = pickle.load(f)

        x_tracks=[]
        y_tracks=[]

        first=True
        for track in tracks:
            x_all_rois = []
            y_all_rois = []
            for roi_points in track :
                if first:
                    xs = roi_points[-1][:,0].tolist()
                    ys = (data_pos["shape"][1]/scale_factor-roi_points[0][:,1]).tolist()
                if len(roi_points)==0:
                    xs  = None
                    ys = None
                else:
                    xs = roi_points[-1][:,0].tolist()
                    ys = (data_pos["shape"][1]/scale_factor-roi_points[-1][:,1]).tolist()
                
                x_all_rois.extend(xs)
                y_all_rois.extend(ys)

            x_tracks.append(x_all_rois)
            y_tracks.append(y_all_rois)

            first=False
            
        points_source.data = dict(x=x_tracks, y=y_tracks)

        point_source.data = dict(
            x=points_source.data['x'][0],
            y=points_source.data['y'][0]
        )

    #_____________________________________________________________
    def invert_axis_callback(attr, old, new) :
        shifts_x = []
        shifts_y = []
        shifts_z = []

        print("invert callback")
        for tp in mp_list:
            tp = str(tp)
            try :
                if 0 in invert_axis_checkboxgroup.active :
                    shifts_x.append(-data_log[tp]["shift_um"]["x"])
                else :
                    shifts_x.append(data_log[tp]["shift_um"]["x"])
                if 1 in invert_axis_checkboxgroup.active :
                    shifts_y.append(-data_log[tp]["shift_um"]["y"])
                else :
                    shifts_y.append(data_log[tp]["shift_um"]["y"])
                if 2 in invert_axis_checkboxgroup.active :
                    shifts_z.append(-data_log[tp]["shift_um"]["z"])
                else :
                    shifts_z.append(data_log[tp]["shift_um"]["z"])
            except KeyError:
                print('no time point---',tp,'----')


        shifts_um_cumsum_x = np.cumsum(np.array(shifts_x),axis=0)
        shifts_um_cumsum_y = np.cumsum(np.array(shifts_y),axis=0)
        shifts_um_cumsum_z = np.cumsum(np.array(shifts_z),axis=0)

        shift_mu_source.data = dict(x=shifts_x, y=shifts_y, z=shifts_z, t=[i for i in range(0,len(shifts_x))])
        shifts_text.text = (
            f"x (&micro;m): {shift_mu_source.data["x"][slider.value]:.2f}<br>"
            f"y (&micro;m): {shift_mu_source.data["y"][slider.value]:.2f}<br>"
            f"z (&micro;m): {shift_mu_source.data["z"][slider.value]:.2f}"
        )

        trajectory_source.data = dict(x=shifts_um_cumsum_x, y=shifts_um_cumsum_y, z=shifts_um_cumsum_z, t=np.arange(len(shifts_um_cumsum_x)))
        trajectory_highlight_source.data = {
            "x": [trajectory_source.data["x"][slider.value]],
            "y": [trajectory_source.data["y"][slider.value]],
            "z": [trajectory_source.data["z"][slider.value]],
        }

    invert_axis_checkboxgroup.on_change("active", invert_axis_callback)
    #_______________________________________________________
    def reload_images():
        load_images(status.text.replace("Selected folder: ",""), True)
        load_tracking(status.text.replace("Selected folder: ",""), True)
    btn_reload = Button(label="Reload", button_type="success")
    btn_reload.on_click(reload_images)

   #___________________________________________________________________________________________
    def update_contrast(attr, old, new):
        low, high = new 
        color_mapper.low = low
        color_mapper.high = high

    contrast_slider = RangeSlider(start=0, end=255, value=(0, 255), step=1, title="Contrast", width=150)
    contrast_slider.on_change('value', update_contrast)



    #_______________________________________________________
    def save_movie():
        images=images_source.data['image']
        for i in range(len(images)):
            images[i] = np.flip(images[i], axis=0)
        rois_per_frame=[]
        for i in range(len(rects_source.data['x'])):
            x = rects_source.data['x'][i]
            y = rects_source.data['y'][i]
            width = rects_source.data['width'][i]
            height = rects_source.data['height'][i]
            local_rois=[]
            for j in range(len(x)):
                x_val = x[j]
                y_val = images[0].shape[0]-y[j]
                width_val = width[j]
                height_val = height[j]
                if width_val > 0 and height_val > 0:
                    local_rois.append((x_val - width_val / 2., y_val - height_val / 2., x_val + width_val / 2., y_val + height_val / 2.))
            rois_per_frame.append(local_rois)
        points=[]
        for i in range(len(points_source.data['x'])):
            x = points_source.data['x'][i]
            y = [images[0].shape[0]-points_source.data['y'][i][j] for j in range(len(points_source.data['y'][i]))]
            pts = list(zip(x, y))
            points.append(pts)
        frames = []
        for i, (img_array, rois, pts) in enumerate(zip(images, rois_per_frame, points)):
            img = Image.fromarray(img_array).convert("RGB")
            draw = ImageDraw.Draw(img)
            for roi in rois:
                draw.rectangle(roi, outline="red", width=2)
            draw.text((5, 5), f"Frame {i}", fill="white")
            for x, y in pts:
                r = 3
                draw.ellipse((x - r, y - r, x + r, y + r), fill="red")
           
            frames.append(img)

        name="{}/{}.gif".format(status.text.replace("Selected folder: ",""), dropdown_position.value.replace(" ", "_"))
        print(os.path.split(status.text.replace("Selected folder: ","")))
        print(name)
        print("Saved  movie as ",name)
        button_save.label = "Save movie"
        button_save.button_type = "success"
        frames[0].save(name, save_all=True, append_images=frames[1:], duration=100, loop=0)
        #status.text = "Movie saved as timelapse_multi_rois.gif"

        #_______________________________________________________
    def save_movie_short():
        global detect_model
        button_save.label = "Processing"
        button_save.button_type = "danger"
  
        curdoc().add_next_tick_callback(save_movie)

    button_save = Button(label="Save movie", button_type="success")
    button_save.on_click(save_movie_short)

    #_______________________________________________________
    def open_file_dialog():
        try:
            root = tk.Tk()
            root.attributes('-topmost', True)
            root.withdraw()
            root.update()
            dir_path = filedialog.askdirectory(parent=root)
            root.destroy()

            if dir_path:
                status.text = f"Selected folder: {dir_path}"
                pos_list_full=glob.glob(os.path.join(dir_path, "*", "embryo_tracking"))
                pos_list = []
                for p in pos_list_full:
                    pos_list.append(os.path.split(os.path.split(p)[0])[-1])

                dropdown_position.options = pos_list

                #load_images(dir_path)
                #load_tracking(dir_path)
                #slider.value=0
            else:
                status.text = "No directory selected."
        except Exception as e:
            status.text = f"Error: {e}"

    #_______________________________________________________
    def update_pos(attr, old, new):
        
        dir_path = os.path.join(status.text.replace("Selected folder: ",""), new )
        print("dir path ",dir_path)
        load_images(dir_path)
        load_tracking(dir_path)
        slider.value=0
    dropdown_position.on_change('value', update_pos)

    select_button = Button(label="Browse Folder...", button_type="primary")
    select_button.on_click(open_file_dialog)

    select_layout = row(mk_div(), select_button, mk_div(), dropdown_position)
    slider_layout = row(mk_div(), slider)
    text_layout = row(mk_div(), status)
    reload_layout = row(mk_div(), btn_reload, button_save)
    
    left_col  = column(select_layout, p_img, slider_layout, reload_layout, text_layout, row(mk_div(),contrast_slider))
    trajectory_row = row(p_trajectory_xy, p_trajectory_xz, p_trajectory_yz)
    shifts_row = row(p_shifts, shifts_text, invert_axis_checkboxgroup)
    right_col = column(mk_div(), shifts_row, trajectory_row)
    layout = row(left_col, right_col)

    return layout



#_______________________________________________________
def make_document(doc):
    layout = make_layout()
    doc.title = 'Tracking visualisation'
    doc.add_root(layout)



#_______________________________________________________
def run_server():
    io_loop = IOLoop()
    server = Server({'/': make_document},
                    io_loop=io_loop,
                    allow_websocket_origin=["localhost:5007"],
                    port=5007)
    server.start()
    print("Starting Bokeh server on http://localhost:5007/")
    server.io_loop.start()
    #to run detached
    #io_loop.start()



if __name__ == '__main__':
    run_server()
    #to run a detached server
    #import threading
    #thread = threading.Thread(target=run_server)
    #thread.start()
    #print("Bokeh is now serving at http://localhost:5006/")
