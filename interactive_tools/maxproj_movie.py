import os, sys
import tifffile
import numpy as np  
import glob
from PIL import Image, ImageDraw

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
def create_images(out_dir, img_list):
    for img in img_list:
        print("processing image ",img)
        out_name = os.path.join(out_dir,os.path.split(img)[1].replace('.tif', '_maxproj.tif'))
        if os.path.exists(out_name):
            print("skipping existing ", out_name)
            continue
        im =  tifffile.imread(img)
        arr = np.array(im)
        arr_down = downscale_image(arr, n=2)
        max_proj = np.max(arr_down, axis=0)
        print("-------out name:  ",out_name)
        tifffile.imwrite(out_name, max_proj)


#_______________________________________________________
def save_movie(out_dir):
    img_list = glob.glob(os.path.join(out_dir,"*_maxproj.tif"))

    out_name=os.path.join(out_dir, "maxproj_movie.gif")

    frames = []
    for i, img in enumerate(img_list):
        img_array = tifffile.imread(img)
        max_value = np.max(img_array)
        min_value = np.min(img_array)
        max_proj_norm = (img_array - min_value)/(max_value-min_value)*255
        max_proj_norm = max_proj_norm.astype(np.uint8)
        img = Image.fromarray(max_proj_norm).convert("RGB")
        draw = ImageDraw.Draw(img)    
        draw.text((5, 5), f"Frame {i}", fill="white")
        frames.append(img)
    frames[0].save(out_name, save_all=True, append_images=frames[1:], duration=100, loop=0)
    print("Saved  movie as ",out_name)

    import moviepy as mp

    clip = mp.VideoFileClip(out_name)
    clip.write_videofile(out_name.replace(".gif",".mp4"))


#_______________________________________________________
if __name__ == "__main__":
    if len(os.sys.argv)>2:
        base_path = os.path.abspath(os.sys.argv[1])
        channel = os.sys.argv[2]  
    else:
        print("No path provided, run as python maxproj_movie.py <path_to_data>")
        sys.exit(0)

    img_list = glob.glob(os.path.join(base_path,"*{}.tif".format(channel)))
    print("---- img_list ", img_list)
    out_dir = os.path.join(base_path, "maxproj_notracking")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    create_images(out_dir, img_list)
    save_movie(out_dir)