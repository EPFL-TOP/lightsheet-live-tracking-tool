import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import os, glob, sys
import tifffile
import numpy as np
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("No path provided, run as python draw_ROI_onmaxproj.py <path_to_data>")
        sys.exit(0)

    image_list = glob.glob(os.path.join(image_path,"embryo_tracking", "max_proj", "*_max_proj.tif"))
    json_path = os.path.join(image_path, "embryo_tracking", "logs.json")

    if not os.path.exists(json_path):
        print(f"Warning: No JSON file found at {json_path}")
        sys.exit(0)
    with open(json_path, 'r') as f:
        log_data = json.load(f)

    image_list = sorted(image_list)
    first = False


    for clip in [True, False]:
        for dashed in [True, False]:

            for img in image_list:
                print("Processing image: ", img)
                image = tifffile.imread(img)


                h, w = image.shape

                if clip:
                    frames = np.stack(image)
                    q1, q99 = np.quantile(frames, [0.0001, 0.9999])
                    print(f"Quantile 1%: {q1}, Quantile 99%: {q99}")
                    value_range = q99 - q1
                    normalized_image = np.clip((frames - q1) / value_range, 0, 1)
                    image = normalized_image


                time_point = int(os.path.basename(img).replace('.tif','').split('t')[-1].split('_max_proj')[0])

                print(f"Timepoint: {time_point}")

                ROI = log_data[str(time_point)]['roi'][0]  # Assuming there's only one ROI, adjust if there are multiple
                x, y, width, height = ROI['x'], ROI['y'], ROI['width'], ROI['height']



                dpi = 100
                fig = plt.figure(figsize=(w/dpi, h/dpi), dpi=dpi)

                ax = plt.axes([0,0,1,1])   # remove margins
                ax.imshow(image, cmap="gray")
                ax.axis("off")
                if dashed:          
                    rect = patches.Rectangle((x - width/2, y - height/2), width, height, linewidth=3, edgecolor='r', facecolor='none', linestyle="--")
                else:
                    rect = patches.Rectangle((x - width/2, y - height/2), width, height, linewidth=3, edgecolor='r', facecolor='none')


                if first:
                    # ---- SCALE BAR ----
                    pixel_size = 4 * 0.347  # µm / pixel
                    scale_um = 100

                    scale_pixels = scale_um / pixel_size

                    fontprops = fm.FontProperties(size=14)

                    scalebar = AnchoredSizeBar(
                        ax.transData,
                        scale_pixels,
                        f"{scale_um} µm",
                        "lower right",
                        pad=0.4,
                        color="white",
                        frameon=False,
                        size_vertical=3,
                        fontproperties=fontprops
                    )

                    ax.add_artist(scalebar)
                    first=False
                    to_add ='_clipped' if clip else ''
                    to_add += '_dashed' if dashed else ''
            
                    plt.savefig(img.replace(".tif",f"{to_add}.png"), dpi=dpi)
                    #plt.close()
                ax.add_patch(rect)

                to_add ='_ROI_clipped' if clip else '_ROI'
                to_add += '_dashed' if dashed else ''
            
                plt.savefig(img.replace(".tif",f"{to_add}.png"), dpi=dpi)
                plt.close()


