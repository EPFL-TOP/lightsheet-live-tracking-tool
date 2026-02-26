import tkinter as tk
from tornado.ioloop import IOLoop
from tkinter import filedialog
from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, BoxEditTool, TapTool, LabelSet, Button, CheckboxGroup, TextInput, Div, Range1d, Slider, Select, RangeSlider, LinearColorMapper
from bokeh.layouts import column, row
from bokeh.events import SelectionGeometry
from bokeh.server.server import Server
import numpy as np
import json, os, pathlib
import tifffile
import glob, sys, random, shutil

#_______________________________________________________
def make_document(doc):

    initial_img = np.random.randint(0, 255, (1000, 1000), dtype=np.uint8)[::-1]
    initial_img_white = np.ones((100, 100))*254

    status = Div(text="")
    slider = Slider(start=0, end=0, value=0, step=1, title="time frame", width=700)

    image_source  = ColumnDataSource(data=dict(image=[initial_img], x=[0], y=[0], dw=[initial_img.shape[1]], dh=[initial_img.shape[0]]))
    images_source = ColumnDataSource(data=dict(image=[initial_img_white], x=[0], y=[0], dw=[initial_img_white.shape[1]], dh=[initial_img_white.shape[0]]))
    rect_source   = ColumnDataSource(data=dict(x=[], y=[], width=[], height=[], index=[], label_x=[], label_y=[]))

    image_exist_source  = ColumnDataSource(data=dict(image=[initial_img_white], x=[0], y=[0], dw=[initial_img_white.shape[1]], dh=[initial_img_white.shape[0]]))
    rect_exist_source   = ColumnDataSource(data=dict(x=[], y=[], width=[], height=[]))

    p_img = figure(
        title="ROI selector CNN training",
        x_range=(0, initial_img.shape[1]), y_range=(0, initial_img.shape[0]),
        tools="pan,wheel_zoom,box_select,reset,undo,redo",
        match_aspect=True, 
        width=800, height=800
    )

    p_img_exist = figure(
        title="Already annotated?",
        x_range=(0, initial_img_white.shape[1]), y_range=(0, initial_img_white.shape[0]),
        tools="pan,wheel_zoom,box_select,reset,undo,redo",
        match_aspect=True, 
        width=500, height=500
    )

    color_mapper = LinearColorMapper(palette="Greys256", low=0, high=255)
    p_img.image('image', x='x', y='y', dw='dw', dh='dh', source=image_source,  color_mapper=color_mapper)
    p_img_exist.image('image', x='x', y='y', dw='dw', dh='dh', source=image_exist_source,  color_mapper=color_mapper)

    rect_glyph = p_img.rect(
        'x', 'y', 'width', 'height', source=rect_source,
        fill_alpha=0.2, fill_color='blue', line_color='red', line_width=2
    )

    rect_glyph_exist = p_img_exist.rect(
        'x', 'y', 'width', 'height', source=rect_exist_source,
        fill_alpha=0.2, fill_color='blue', line_color='red', line_width=2
    )

    box_edit = BoxEditTool(renderers=[rect_glyph], num_objects=100)
    p_img.add_tools(box_edit)
    p_img.toolbar.active_drag = box_edit

    tap = TapTool(renderers=[rect_glyph])
    p_img.add_tools(tap)
    p_img.toolbar.active_tap = tap


     #_______________________________________________________
    def get_image_name(file_name):
        outdir = r'E:\tail_tracking'
        outfile_name = file_name.replace('/',"_").replace("\\","_").replace(":","").replace(" ","_")
        return (outdir, outfile_name)

    #_______________________________________________________
    def save_rectangles():
        file_name = images_source.data['name'][slider.value]
        outdir, outfile_name = get_image_name(file_name)
        rand=random.uniform(0,1)
        if rand>0.2: outdir=os.path.join(outdir,'train')
        else: outdir=os.path.join(outdir,'valid')

        data = rect_source.data
        out = []

        for i, (x, y, w, h) in enumerate(zip(data.get('x', []), data.get('y', []), data.get('width', []), data.get('height', []))):
            out.append({'x': x, 'y': image_source.data['image'][0].shape[1] - y, 'width': w, 'height': h})

        print('outfile_name name ', outfile_name)

        outdict = {'RoIs':out,  'image':outfile_name}

        print('     ',outfile_name, '  ', os.path.join(outdir, outfile_name))
        shutil.copyfile(file_name, os.path.join(outdir, outfile_name))

        outjson=os.path.join(outdir,outfile_name.replace(".tif",".json"))
        with open(outjson, "w") as f:
            json.dump(outdict, f, indent=2)
        print("Saved: ",outjson)
        check_existing_image(file_name)
    btn_save = Button(label="Save region", button_type="success")
    btn_save.on_click(save_rectangles)


    #_______________________________________________________
    def delete_selected():
        inds = rect_source.selected.indices
        print('inds   ',inds)
        if not inds:
            return
        data = dict(rect_source.data)
        for i in sorted(inds, reverse=True):
            for key in data:
                data[key].pop(i)
        rect_source.data = data
        rect_source.selected.indices = []

    btn_delete = Button(label="Delete Selected", button_type="danger")
    btn_delete.on_click(delete_selected)

    #___________________________________________________________________________________________
    def check_existing_image(imgname):

        outdir, image_name = get_image_name(imgname)
        exist=False
        for d in ["train", "valid"]:
            f=os.path.join(outdir,d,image_name)
            if os.path.exists(f):
                exist=True
                im =  tifffile.imread(f)
                max_value = np.max(im)
                min_value = np.min(im)
                max_proj_norm = (im - min_value)/(max_value-min_value)*255
                max_proj_norm = max_proj_norm.astype(np.uint8)
                max_proj_norm = np.flip(max_proj_norm,0)

                with open(f.replace('.tif','.json'), 'r') as file:                    
                    data_pos = json.load(file)

                x=[]
                y=[]
                h=[]
                w=[]
                for roi in data_pos['RoIs']:
                    x.append(roi['x'])
                    y.append(im.shape[1]-roi['y'])
                    w.append(roi['width'])
                    h.append(roi['height'])

                image_exist_source.data =  {'image':[max_proj_norm], 
                                    'x':[0],
                                    'y':[0],
                                    'dw':[im.shape[1]],
                                    'dh':[im.shape[0]]
                                    }
                

                rect_exist_source.data = dict(x=x, y=y, height=h, width=w)

        if not exist:
            image_exist_source.data =  {'image':[initial_img_white], 
                'x':[0],
                'y':[0],
                'dw':[initial_img_white.shape[1]],
                'dh':[initial_img_white.shape[0]]
                }
            rect_exist_source.data = dict(x=[], y=[], height=[], width=[])


    #___________________________________________________________________________________________
    def callback_slider(attr, old, new):
        time_point = slider.value
        image_source.data = {'image':[images_source.data['image'][time_point]], 
                            'x':[images_source.data['x'][time_point]],
                            'y':[images_source.data['y'][time_point]],
                            'dw':[images_source.data['dw'][time_point]],
                            'dh':[images_source.data['dh'][time_point]],
                            'name':[images_source.data['name'][time_point]]}
        
        check_existing_image(images_source.data['name'][time_point])


    slider.on_change('value', callback_slider)


    #_______________________________________________________
    def next_image():
        n_img = slider.end+1
        current_index = slider.value
        current_index = (current_index + 1) % n_img
        if current_index>slider.end:current_index=slider.start
        if current_index<slider.start:current_index=slider.end
        slider.value = current_index

    btn_next = Button(label="Next")
    btn_next.on_click(next_image)

    #_______________________________________________________
    def previous_image():
        n_img = slider.end+1
        current_index = slider.value
        current_index = (current_index - 1) % n_img
        if current_index>slider.end:current_index=slider.start
        if current_index<slider.start:current_index=slider.end
        slider.value = current_index

    btn_prev = Button(label="Previous")
    btn_prev.on_click(previous_image)

    #_______________________________________________________
    def select_roi_callback(event):
        if isinstance(event, SelectionGeometry):
            if event.geometry["type"]!='rect':return

            data_rect = dict(
                x= rect_source.data['x']+[event.geometry['x0'] + (event.geometry['x1']-event.geometry['x0'])/2. ],
                y= rect_source.data['y']+[event.geometry['y0'] + (event.geometry['y1']-event.geometry['y0'])/2.],
                width= rect_source.data['width']+[event.geometry['x1']-event.geometry['x0']],
                height= rect_source.data['height']+[event.geometry['y1']-event.geometry['y0']],
                index=rect_source.data['index']+["none"],
                label_x=rect_source.data['label_x']+[event.geometry['x0']],
                label_y=rect_source.data['label_y']+[event.geometry['y0']]
                )

            rect_source.data = data_rect

            print( rect_source.data)

    p_img.on_event(SelectionGeometry, select_roi_callback)


#_______________________________________________________
    def load_images(dir_path):
        #out_name = os.path.join(dir_path,"embryo_tracking","max_proj")
        out_name = dir_path
        img_list=glob.glob(os.path.join(out_name,"*.tif"))
        print('nb images max proj: ',len(img_list))
        images_ds=[]
        x_ds=[]
        y_ds=[]
        dw_ds=[]
        dh_ds=[]
        images_name=[]

        for idx, img in enumerate(img_list):

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
            images_name.append(img)
        x_range = Range1d(start=0, end=images_ds[0].shape[0])
        y_range = Range1d(start=0, end=images_ds[0].shape[1])
        x_range_2 = Range1d(start=0, end=images_ds[0].shape[0])
        y_range_2 = Range1d(start=0, end=images_ds[0].shape[1])
        p_img.x_range=x_range
        p_img.y_range=y_range
        p_img_exist.y_range=y_range_2
        p_img_exist.x_range=x_range_2
        images_source.data = dict(image=images_ds, x=x_ds, y=y_ds, dw=dw_ds, dh=dh_ds,name=images_name)
        image_source.data = dict(image=[images_ds[0]], x=[x_ds[0]], y=[y_ds[0]], dw=[dw_ds[0]], dh=[dh_ds[0]], name=[images_name[0]])
        check_existing_image(images_name[0])
        slider.end = len(images_ds)-1

    #_______________________________________________________
    def open_file_dialog():
        try:
            root = tk.Tk()

            # make it topmost, withdraw it so you don’t actually see the empty window
            root.attributes('-topmost', True)
            root.withdraw()
            root.update()  # sometimes needed to actually apply the topmost flag

            # show the dialog, telling it that our root is the parent  
            dir_path = filedialog.askdirectory(parent=root)

            root.destroy()

            if dir_path:
                status.text = f"Selected folder: {dir_path}"
                load_images(dir_path)
                slider.value=0
            else:
                status.text = "No directory selected."
        except Exception as e:
            status.text = f"Error: {e}"
    select_button = Button(label="Browse Folder...", button_type="primary")
    select_button.on_click(open_file_dialog)

    #_______________________________________________________
    def mk_div(**kwargs):
        return Div(text='<div style="background-color: white; width: 20px; height: 1px;"></div>', **kwargs)


    #_______________________________________________________
    def use_same_rect(attr, old, new):
        if 0 in same_rect.active:
            rect_exist_source.data = dict(x=rect_source.data['x'], y=rect_source.data['y'], 
                                          height=rect_source.data['height'], width=rect_source.data['width'])
        else:
            rect_source.data = dict(x=[], y=[], height=[], width=[])
    same_rect = CheckboxGroup(labels=["Use same rectangles"], active=[], width=200)
    same_rect.on_change('active', use_same_rect)


    select_layout = row(mk_div(), select_button)
    slider_layout = row(mk_div(), slider)
    next_prev_layout = row(mk_div(), btn_prev, btn_next, mk_div(), same_rect)
    text_layout = row(mk_div(), status)
    save_layout = row(mk_div(), btn_save, mk_div(), btn_delete)
 
    left_col  = column(select_layout, p_img, slider_layout,next_prev_layout, save_layout, text_layout)
    right_col = column(mk_div(), p_img_exist)
    layout = row(left_col, mk_div(), right_col)

    doc.title = 'Annotation'
    doc.add_root(layout)

#_______________________________________________________
def run_server():
    io_loop = IOLoop()
    server = Server({'/': make_document},
                    io_loop=io_loop,
                    allow_websocket_origin=["localhost:5010"],
                    port=5010)
    server.start()
    print("Starting Bokeh server on http://localhost:5010/")
    server.io_loop.start()


#_______________________________________________________
if __name__ == '__main__':
    run_server()