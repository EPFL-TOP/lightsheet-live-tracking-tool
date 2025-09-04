import tkinter as tk
from tornado.ioloop import IOLoop
from tkinter import filedialog
from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, BoxEditTool, TapTool, LabelSet, Button, CheckboxGroup, PolyDrawTool, TextInput, Div, Range1d, Slider, Select, RangeSlider, LinearColorMapper, FileInput, PreText
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from  training_tools import tail_detection_visu as tdv
except ModuleNotFoundError: 
    print("Module 'tail_detection_visu' not found. Ensure the training_tools package is installed or available in the PYTHONPATH.")

try:
    from  tracking_tools.utils import tracking_utils as tutils
except ModuleNotFoundError: 
    print("Module 'tail_detection_visu' not found. Ensure the training_tools package is installed or available in the PYTHONPATH.")


#_______________________________________________________
def make_document(doc):
    #### SHARED VARIABLES ######
    updating = False
    detect_model = None

    ####### GENERAL WIDGETS ##########
    # Checkboxes
    checkbox_maxproj = CheckboxGroup(labels=["Max projection"], active=[])
    checkbox_detection = CheckboxGroup(labels=["Use detection"], active=[])
    
    # Dropdowns
    downscale=['0','1','2','3','4','5']
    dropdown_downscale  = Select(value=downscale[0], title='Downscaling', options=downscale)
    try:
        base_path = os.path.dirname(__file__)
    except NameError:
        base_path = os.path.dirname(os.path.abspath(sys.argv[0]))
        
    default_model_path=os.path.abspath(os.path.join(base_path, "..", "weights"))
    model_detect=[]
    if os.path.isdir(default_model_path):
        models = glob.glob(os.path.join(default_model_path,'*.pth'))
        model_detect=[os.path.split(model)[-1].replace('.pth','') for model in models]
    dropdown_model  = Select(value=downscale[0], title='Detect Model', options=model_detect)

    # Sliders
    contrast_slider = RangeSlider(start=0, end=255, value=(0, 255), step=1, title="Contrast", width=150)
    slice_slider = Slider(start=0, end=0, value=0, step=1, title="z-slice", width=250)
    mask_alpha_slider = Slider(start=0, end=1, value=0, step=0.01, title="Mask opacity", width=200)
    points_alpha_slider = Slider(start=0, end=1, value=0, step=0.01, title="Points opacity", width=200)

    # Buttons
    btn_save = Button(label="Save Tracking RoIs", button_type="success")
    btn_down = Button(label="Move Down")
    btn_up = Button(label="Move Up")
    btn_delete = Button(label="Delete Selected", button_type="danger")
    select_image_button = Button(label="Browse Image...", button_type="primary")
    select_model_button = Button(label="Browse Model Folder...", button_type="primary")
    detect_button = Button(label="Run detect model", button_type="primary")

    # Fileinputs
    file_input = FileInput()

    # Texts
    status = Div(text="")
    model_status = Div(text=f"Selected model path: {default_model_path}")
    


    ############ DATA SOURCES #############
    # Initial dummy image
    initial_img = np.random.randint(0, 255, (10, 1000, 1000), dtype=np.uint8)[::-1]

    # Loaded (original) image datasource
    original_source = ColumnDataSource(data=dict(
        image=[initial_img], x=[0], y=[0], dw=[initial_img.shape[2]], dh=[initial_img.shape[1]]
    ))

    # Displayed image data source 
    displayed_source = ColumnDataSource(data=dict(
        image=[initial_img[0]], x=[0], y=[0], dw=[initial_img.shape[1]], dh=[initial_img.shape[0]]
    ))

    # Working image data source (image used for processings)
    working_source = ColumnDataSource(data=dict(
        image=[initial_img[0]], x=[0], y=[0], dw=[initial_img.shape[1]], dh=[initial_img.shape[0]]
    ))

    # Maximum projection data source
    maxproj_source = ColumnDataSource(data=dict(
        image=[initial_img[0]], x=[0], y=[0], dw=[initial_img.shape[1]], dh=[initial_img.shape[0]]
    ))

    # Mask data source
    mask_rgba_source = ColumnDataSource(data=dict(
        image=[], x=[], y=[], dw=[], dh=[], alpha=[]
    ))

    # Points data source
    points_source = ColumnDataSource(data=dict(
        x=[], y=[], alpha=[], color=[], radius=[]
    ))

    # Selection and detection rectangles setup
    select_rectangle_source = ColumnDataSource(data=dict(
        x=[], y=[], width=[], height=[], index=[], label_x=[], label_y=[]
    ))

    detect_rectangle_source = ColumnDataSource(data=dict(
        x=[], y=[], width=[], height=[], score=[], label_x=[], label_y=[]
    ))



    ########## FIGURE SETUP ############
    p = figure(
        title="RoIs tracking selector",
        x_range=(0, initial_img.shape[1]), y_range=(0, initial_img.shape[0]),
        tools="pan,wheel_zoom,box_select,reset,undo,redo",
        match_aspect=True,
        width=800, height=800

    )

    # Display image from source
    color_mapper = LinearColorMapper(palette="Greys256", low=0, high=255)
    p.image('image', x='x', y='y', dw='dw', dh='dh', source=displayed_source,  color_mapper=color_mapper)

    labels = LabelSet(
        x='label_x', y='label_y', text='index', source=select_rectangle_source,
        text_baseline='middle', text_align='left', text_color='white'
    )

    labels_detect = LabelSet(
        x='label_x', y='label_y', text='score', source=detect_rectangle_source,
        text_baseline='middle', text_align='left', text_color='white'
    )

    # RGBA mask for threshold overlay
    mask_glyph = p.image_rgba(
        "image", "x", "y", "dw", "dh", alpha="alpha", source=mask_rgba_source
    )


    rect_glyph_select = p.rect(
        'x', 'y', 'width', 'height', source=select_rectangle_source,
        fill_alpha=0.2, fill_color='blue', line_color='red', line_width=2
    )

    rect_glyph_detect = p.rect(
        'x', 'y', 'width', 'height', source=detect_rectangle_source,
        line_color='white', line_width=2, fill_alpha=0
    )

    # Points overlay
    points_glyph = p.circle(
        "x", "y", "radius",
        fill_alpha = "alpha", 
        line_alpha = "alpha",
        selection_alpha= "alpha",
        nonselection_alpha = "alpha", 
        color = "color", 
        source=points_source,
    )

    p.add_layout(labels)
    p.add_layout(labels_detect)

    # Edit tools
    box_edit = BoxEditTool(renderers=[rect_glyph_select], num_objects=100)
    p.add_tools(box_edit)
    p.toolbar.active_drag = box_edit

    tap = TapTool(renderers=[rect_glyph_select])
    p.add_tools(tap)
    p.toolbar.active_tap = tap


    polygone_source = ColumnDataSource(data=dict(xs=[], ys=[]))

    polygone_renderer = p.patches("xs", "ys", source=polygone_source,
                        fill_alpha=0.9, fill_color="lightblue",
                        line_color="black", line_width=7)

    polygone_draw = PolyDrawTool(renderers=[polygone_renderer], drag=False)
    p.add_tools(polygone_draw)
    p.toolbar.active_tap = polygone_draw
    ############## CORE FUNCTIONS ##############

    def draw_polygone(attr, old, new) :
        print("in draw_polygone")
        d = polygone_source.data
        if d["xs"] == [] :
            return
        xs = d["xs"][-1]
        ys = d["ys"][-1]
        print(xs, ys)
    #    polygone_source.data = dict(xs=[xs], ys=[ys])
    #    print(xs, ys)
    polygone_source.on_change('data', draw_polygone)

    def debug_event(attr, old, new):
        print(f"Change detected: {attr}")
        print("old:", old)
        print("new:", new)

    polygone_source.on_change("data", debug_event)



    def on_geom(event):
        print("Geometry event:", event.geometry)

    p.on_event(SelectionGeometry, on_geom)

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
    def binary2rgba(binary, color=(0, 255, 255), alpha=255):
        h, w = binary.shape
        r, g, b = color
        rgba = np.zeros((h, w), dtype=np.uint32)

        # Fill RGBA Mask :  alpha     red     green    blue
        #                  xxxxxxxx xxxxxxxx xxxxxxxx xxxxxxxx 
        rgba[binary.astype(bool)] = (alpha << 24) | (r << 16) | (g << 8) | b
        return np.flip(rgba, axis=0)
    
    
    #_______________________________________________________________________________________________
    def update_working(attr, old, new) :
        original = original_source.data["image"][0]
        # Maximum porojection
        if 0 in checkbox_maxproj.active :
            working = np.max(original, axis=0)
        else :
            slice_nb = slice_slider.value
            working = original[slice_nb]
        # Downscale image
        scaling_factor = 2 ** int(dropdown_downscale.value)
        if scaling_factor > 1 :
            working = downscale_image(working, scaling_factor)

        # Update data source
        working_source.data = dict(
            image=[working], x=[0], y=[0], dw=[working.shape[1]], dh=[working.shape[0]]
        )



    #_______________________________________________________________________________________________
    def update_display(attr, old, new) :
        print("in update_display")
        working = working_source.data["image"][0]
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

    
    #_______________________________________________________________________________________________
    def update_mask_and_points(attr, old, new) :
        print("in NEW mask and point update")
        working = working_source.data["image"][0]
        d = select_rectangle_source.data

        # Only runs if rectangles are selected
        if d["x"] == [] :
            # Clear data sources
            points_source.data = dict(
                x=[], y=[], color=[], fill_alpha=[], line_alpha=[], radius=[],
            )
            mask_rgba_source.data = dict(
                image=[], x=[], y=[], dw=[], dh=[]
            )
            return

        xs, ys = [], []
        mask = None
        for x, y, width, height, index in zip(d["x"], d["y"], d["width"], d["height"], d["index"]) :
            center_point = (working.shape[0] - y, x) # Flip rectangles y coordinates (because bokeh coordinate system flips y)
            hws = (height/2, width/2)
            points, mask = tutils.generate_uniform_grid_in_region(working, center_point, hws, return_mask=True, grid_size=40) ### TODO : Dynamic kernel size
            points[:,1] = working.shape[0] - points[:,1] # Re-flip y coordinates for display
            xs.extend(points[:,0].tolist())
            ys.extend(points[:,1].tolist())
        
        n_new = len(xs)

        # Update point source
        points_source.data = dict(
            x=xs,
            y=ys,
            color=["red"] * n_new,
            fill_alpha=[1] * n_new,
            line_alpha=[1] * n_new,
            radius=[1] * n_new,
        )

        # Update mask source
        mask_rgba = binary2rgba(mask)
        mask_rgba_source.data = dict(
            image=[mask_rgba], x=[0], y=[0], dw=[mask_rgba.shape[1]], dh=[mask_rgba.shape[0]]
        )


    #________________________________________________________________________________________________
    def update_mask(attr, old, new) :
        working = working_source.data["image"][0]
        base_kernel = 41
        scaling_factor = 2 ** int(dropdown_downscale.value)
        mask = tutils.filter_and_threshold(working, gaussian_kernel=base_kernel//scaling_factor)
        mask_rgba = binary2rgba(mask)
        alpha_slider_value = float(mask_alpha_slider.value)
        mask_rgba_source.data = dict(
            image=[mask_rgba], x=[0], y=[0], dw=[mask_rgba.shape[1]], dh=[mask_rgba.shape[0]], alpha=[alpha_slider_value]
        )

    #_______________________________________________________________________________________________
    def update_mask_alpha(attr, old, new) :
        alpha_slider_value = float(mask_alpha_slider.value)
        mask_rgba_source.data["alpha"] = [alpha_slider_value]

    #________________________________________________________________________________________________
    def update_points(attr, old, new) :
        working = working_source.data["image"][0]
        d = select_rectangle_source.data

        # Only runs if rectangles are selected
        if d["x"] == [] :
            # Clear data sources
            points_source.data = dict(
                x=[], y=[], color=[], fill_alpha=[], line_alpha=[], radius=[],
            )
            return

        base_kernel = 41
        scaling_factor = 2 ** int(dropdown_downscale.value)
        xs, ys = [], []
        for x, y, width, height, index in zip(d["x"], d["y"], d["width"], d["height"], d["index"]) :
            center_point = (working.shape[0] - y, x) # Flip rectangles y coordinates (because bokeh coordinate system flips y)
            hws = (height/2, width/2)
            points = tutils.generate_uniform_grid_in_region(working, center_point, hws, return_mask=False, grid_size=40, gaussian_kernel=base_kernel//scaling_factor) ### TODO : Dynamic kernel size
            points[:,1] = working.shape[0] - points[:,1] # Re-flip y coordinates for display
            xs.extend(points[:,0].tolist())
            ys.extend(points[:,1].tolist())
        
        n_new = len(xs)

        slider_alpha_value = float(points_alpha_slider.value)
        initial_radius = 8
        scaling_factor = 2 ** int(dropdown_downscale.value)
        radius = initial_radius / scaling_factor

        # Update point source
        points_source.data = dict(
            x=xs,
            y=ys,
            color=["red"] * n_new,
            alpha=[slider_alpha_value] * n_new,
            radius=[radius] * n_new,
        )

    #_______________________________________________________________________________________________
    def update_points_alpha(attr, old, new) :
        slider_alpha_value = float(points_alpha_slider.value)
        n_points = len(points_source.data["x"])
        if n_points == 0 :
            points_source.data["alpha"] = []
        else :
            points_source.data["alpha"] = [slider_alpha_value] * n_points


    #_______________________________________________________________________________________________
    def update_original(arr) :
        # Gets 3D array
        original_source.data = dict(
            image=[arr], x=[0], y=[0], dw=[arr.shape[2]], dh=[arr.shape[1]]
        )
        slice_slider.value = 0
        # Set the maximum slider value to the number of slices
        slice_slider.end = arr.shape[0] - 1

        # Remove the rectangles
        detect_rectangle_source.data = dict(
            x=[], y=[], width=[], height=[], score=[], label_x=[], label_y=[]
        )
        select_rectangle_source.data = dict(
            x=[], y=[], width=[], height=[], score=[], label_x=[], label_y=[]
        )


    #_____________________________________________________________________________________________
    def downscale_rectangles(scaling_multiplier) :
        data = select_rectangle_source.data
        new_data = dict(
            x=[], y=[], width=[],  height=[], index=data["index"], label_x=[], label_y=[]
        )
        for x, y, width, height, label_x, label_y in zip(data["x"], data["y"], data["width"], data["height"], data["label_x"], data["label_y"]) :
            new_data["x"].append(x * scaling_multiplier)
            new_data["y"].append(y * scaling_multiplier)
            new_data["width"].append(width * scaling_multiplier)
            new_data["height"].append(height * scaling_multiplier)
            new_data["label_x"].append(label_x * scaling_multiplier)
            new_data["label_y"].append(label_y * scaling_multiplier)
        
        # print(select_rectangle_source.data)
        select_rectangle_source.data = new_data
        # print("------------")
        # print(select_rectangle_source.data)

    #______________________________________________________________________________________________
    def update_rectangles(attr, old, new) :
        diff = int(old) - int(new)
        scaling_multiplier = 2 ** diff
        downscale_rectangles(scaling_multiplier)
        detect_rectangle_source.data = dict(
            x=[], y=[], width=[], height=[], score=[], label_x=[], label_y=[]
        )

    #___________________________________________________________________________________________
    def update_contrast(attr, old, new):
        low, high = new 
        color_mapper.low = low
        color_mapper.high = high

    #_______________________________________________________
    def select_roi_callback(event):
        if isinstance(event, SelectionGeometry):
            if event.geometry["type"]!='rect':return

            data_rect = dict(
                x= select_rectangle_source.data['x']+[event.geometry['x0'] + (event.geometry['x1']-event.geometry['x0'])/2. ],
                y= select_rectangle_source.data['y']+[event.geometry['y0'] + (event.geometry['y1']-event.geometry['y0'])/2.],
                width= select_rectangle_source.data['width']+[event.geometry['x1']-event.geometry['x0']],
                height= select_rectangle_source.data['height']+[event.geometry['y1']-event.geometry['y0']],
                index=select_rectangle_source.data['index']+["none"],
                label_x=select_rectangle_source.data['label_x']+[event.geometry['x0']],
                label_y=select_rectangle_source.data['label_y']+[event.geometry['y0']]
                )

            select_rectangle_source.data = data_rect

    #_______________________________________________________
    def update_labels(attr, old, new):
        nonlocal updating
        if updating:
            return
        updating = True
        d = select_rectangle_source.data
        xs, ys, ws, hs = d.get('x', []), d.get('y', []), d.get('width', []), d.get('height', [])
        idxs, lx, ly = [], [], []
        for i, (x, y, w, h) in enumerate(zip(xs, ys, ws, hs)):
            idxs.append(str(i+1))
            lx.append(x - w/2)
            ly.append(y + h/2)
        # assign full dict back to source to trigger UI update
        select_rectangle_source.data = dict(
            x=xs, y=ys, width=ws, height=hs,
            index=idxs, label_x=lx, label_y=ly
        )
        updating = False

    #__________________________________________________________________________________________
    def delete_selected():
        inds = select_rectangle_source.selected.indices
        print('inds   ',inds)
        if not inds:
            return
        data = dict(select_rectangle_source.data)
        for i in sorted(inds, reverse=True):
            for key in data:
                data[key].pop(i)
        select_rectangle_source.data = data
        select_rectangle_source.selected.indices = []


    #______________________________________________________
    def move_up():
        inds = select_rectangle_source.selected.indices
        if len(inds) != 1:
            return
        i = inds[0]
        if i == 0:
            return
        data = dict(select_rectangle_source.data)
        for key in ['x', 'y', 'width', 'height']:
            data[key][i], data[key][i-1] = data[key][i-1], data[key][i]
        select_rectangle_source.data = data
        select_rectangle_source.selected.indices = [i-1]

    #_______________________________________________________
    def move_down():
        inds = select_rectangle_source.selected.indices
        if len(inds) != 1:
            return
        i = inds[0]
        if i == len(select_rectangle_source.data['x']) - 1:
            return
        data = dict(select_rectangle_source.data)
        for key in ['x', 'y', 'width', 'height']:
            data[key][i], data[key][i+1] = data[key][i+1], data[key][i]
        select_rectangle_source.data = data
        select_rectangle_source.selected.indices = [i+1]

    #_______________________________________________________
    def save_rectangles():

        initial_shape = original_source.data["image"][0].shape[1:]
        print('Initial shape : ', initial_shape)
        scaling = 2 ** int(dropdown_downscale.value)
        data = select_rectangle_source.data
        out = []

        print(status.text.split("Selected image: "))
        filename = status.text.split("Selected image: ")[-1]
        dirname  = pathlib.Path(filename).parent.resolve()
        channel = os.path.basename(filename)
        channel = channel.replace(".tif","").split("_")[-1]
        print(dirname, '   ',channel)



        for i, (x, y, w, h) in enumerate(zip(data.get('x', []), data.get('y', []), data.get('width', []), data.get('height', []))):
            out.append({'x': x*scaling, 'y': initial_shape[0] - y*scaling, 'width': w*scaling, 'height': h*scaling, 'order': i+1})

        use_detection = False
        if 0 in checkbox_detection.active: use_detection = True
        outdict = {'channel':channel, 'shape':original_source.data["image"][0].shape, 'RoIs':out, 'detection':use_detection}
        out_dirname = os.path.join(dirname, "embryo_tracking")
        if not os.path.isdir(out_dirname):
            os.mkdir(out_dirname)
        with open(os.path.join(out_dirname,"tracking_RoIs.json"), "w") as f:
            json.dump(outdict, f, indent=2)
        print("Saved: ",os.path.join(out_dirname,"tracking_RoIs.json"))




    #_______________________________________________________
    def choose_model_detect(attr, old, new):
        print('loading model: ', dropdown_model.value)
        nonlocal detect_model
        detect_model = tdv.DetectModel()
        model_path = model_status.text.replace("Selected model path: ","")
        # print(model_path)
        print(os.path.join(model_path,dropdown_model.value+'.pth'))
        detect_model.load_model_detect(os.path.join(model_path,dropdown_model.value+'.pth'), 2, 'cpu')

    #_______________________________________________________
    def test_model_detect_long():
        nonlocal detect_model
        scaling_factor = 2 ** int(dropdown_downscale.value)
        image = original_source.data["image"][0].copy()
        image = np.max(image, axis=0)
        image = downscale_image(image, scaling_factor)
        print(image.shape)
        # image_flip = np.flip(image,0)
        # image_flip_cp = image_flip.copy()
        # tifffile.imwrite('test_flip.tif',image_flip_cp)
        # tifffile.imwrite('test.tif',image)
        # print('flip ',image_flip.shape)
        # print('flimageip ',image.shape)
        image_pp = tdv.preprocess_image_pytorch(image).to(detect_model.device)
        labels = detect_model.get_predictions(image_pp)

        x=[]
        y=[]
        width=[]
        label_x=[]
        label_y=[]
        height=[]
        score=[]
        for idx, box in enumerate(labels[0]['boxes']):
            x_min, y_min, x_max, y_max = box.cpu().numpy()
            score.append(round(labels[0]['scores'][idx].cpu().numpy().tolist(), 3))
            x.append(x_min+(x_max-x_min)/2.)
            y.append(image.shape[1]-(y_min+(y_max-y_min)/2.))
            width.append(y_max-y_min)
            height.append(x_max-x_min)
            label_x.append(x_min+(x_max-x_min)/2)
            label_y.append(image.shape[1]-(y_min+10))

        detect_rectangle_source.data=dict(x=x, y=y, width=width, height=height, score=score, label_x=label_x, label_y=label_y)
        detect_button.label = "Run detect model"
        detect_button.button_type = "primary"
        print(detect_rectangle_source.data)

    #_______________________________________________________
    def test_model_detect():
        nonlocal detect_model
        detect_button.label = "Processing"
        detect_button.button_type = "danger"
  
        curdoc().add_next_tick_callback(test_model_detect_long)

    detect_button = Button(label="Run detect model", button_type="primary")
    detect_button.on_click(test_model_detect)

    output = PreText(text="No file selected")
    select = Select(title="Files in folder:", value=None, options=[])

    _root = tk.Tk()
    _root.withdraw()

    #_______________________________________________________
    def _get_parent():
        win = tk.Toplevel(_root)
        win.overrideredirect(True)
        win.geometry("1x1+200+200")
        win.lift()
        win.attributes("-topmost", True)
        win.focus_force()
        return win

    #_______________________________________________________
    def select_file():
        parent = _get_parent()
        filename = filedialog.askopenfilename(parent=parent)
        parent.destroy()

        if not filename:
            output.text = "No file selected"
            return

        try:
            im = np.array(tifffile.imread(filename))
            # Handle 2D images as 3D images with depth = 1
            if im.ndim == 2 :
                im = im[np.newaxis, ...]
            status.text = f"Selected image: {filename}"
            update_original(im)

        except Exception as e:
            status.text = f"Error loading image: {e}"




    #_______________________________________________________
    def select_folder():
        parent = _get_parent()
        folder = filedialog.askdirectory(parent=parent)
        parent.destroy()

        if folder:
            model_detect=[]
            if os.path.isdir(folder):
                models = glob.glob(os.path.join(folder,'*.pth'))
                model_detect=[os.path.split(model)[-1].replace('.pth','') for model in models]
            dropdown_model.options = model_detect
            model_status.text = f"Selected model path: {folder}"
            
            if len(model_detect)==0:
                model_status.text = f"No models in selected model path: {folder}"

        else:
            status.text = "No directory selected."



    
  

    ########## CALLBACKS #########
    dropdown_model.on_change('value', choose_model_detect)
    select_model_button.on_click(select_folder)
    btn_save.on_click(save_rectangles)
    btn_down.on_click(move_down)
    btn_up.on_click(move_up)
    btn_delete.on_click(delete_selected)
    select_rectangle_source.on_change('data', update_labels, update_points)
    p.on_event(SelectionGeometry, select_roi_callback)
    contrast_slider.on_change('value', update_contrast)
    dropdown_downscale.on_change("value", update_rectangles, update_working)
    select_image_button.on_click(select_file)
    working_source.on_change("data", update_display, update_mask, update_points)
    original_source.on_change("data", update_working)
    slice_slider.on_change("value", update_working)
    checkbox_maxproj.on_change("active", update_working)
    mask_alpha_slider.on_change("value", update_mask_alpha)
    points_alpha_slider.on_change("value", update_points_alpha)



    ########## LAYOUT ##########
    #___________________________________________________________________________________________
    def mk_div(**kwargs):
        return Div(text='<div style="background-color: white; width: 20px; height: 1px;"></div>', **kwargs)
        
    controls = row(mk_div(),btn_up, btn_down, btn_delete, btn_save)
    slider_layout = row(mk_div(),slice_slider)
    status_layout = row(mk_div(), status)
    status_layout2 = row(mk_div(), model_status)


    layout = column(
        mk_div(),           
        row(
            mk_div(),select_image_button,mk_div(), select_model_button, 
        ), 
        row(
            p,column(checkbox_maxproj,dropdown_downscale, contrast_slider, dropdown_model,detect_button, checkbox_detection, mask_alpha_slider, points_alpha_slider)
        ), 
        slider_layout, controls, status_layout,status_layout2)
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
