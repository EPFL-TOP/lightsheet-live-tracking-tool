import numpy as np
import skimage.draw as draw
from skimage.filters import threshold_otsu, threshold_mean
from skimage.filters import gaussian
from skimage.morphology import erosion, disk
from scipy.signal import medfilt

def get_box(shape, center_point, hws) :
    """Generate a mask in a box shape in 2D or 3D

    Args:
        shape (_type_): The original image shape
        center_point (_type_): The center of the box
        hws (_type_): The half window size of the box

    Returns:
        ndarray: image containing the box mask
    """
    shape = np.array(shape)
    ndims = len(shape)
    center_point = np.asarray(center_point, dtype=(int))
    hws = np.asarray(hws, dtype=(int))

    # Compute a full rectangle in the x and y axis
    mask = np.zeros(shape, dtype=bool)
    if ndims == 3 :
        start = (center_point[1]-hws[1], center_point[2]-hws[2])
        stop = (center_point[1]+hws[1], center_point[2]+hws[2])
        rr, cc = draw.rectangle(start, stop)
    else :
        start = (center_point[0]-hws[0], center_point[1]-hws[1])
        stop = (center_point[0]+hws[0], center_point[1]+hws[1])
        rr, cc = draw.rectangle(start, stop)

    # Ensure that the mask does not goes out of bounds
    if ndims == 3:
        top = np.max([center_point[0]-hws[0], 0])
        bottom = np.min([center_point[0]+hws[0], shape[0]])
        valid = (rr >= 0) & (rr < shape[1]) & (cc >= 0) & (cc < shape[2])
    else :
        valid = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])

    # Create the mask  
    if ndims == 3 :
        mask[top:bottom, rr[valid], cc[valid]] = True
    else :
        mask[rr[valid], cc[valid]] = True
    return mask

def crop_image(image, center_point, hws, allow_oob=True) :
    """Crop a 3D or 2D image with a box

    Args:
        image (ndarray): The original image
        center_point (_type_): The center point of the cropping box
        hws (_type_): the half window sozeof the cropping box
        allow_oob (bool, optional): Allow the cropping box to go out of bound of the image. Will pad the croped image with zeros to respect the croping box shape. Defaults to True.

    Returns:
        _type_: The croped image
    """
    # Convert everything to numpy arrays
    center_point = np.array(center_point).astype(int)
    hws = np.array(hws).astype(int)
    shape = np.array(image.shape)

    # Get the start and end of the cropping region
    start = center_point - hws
    end = center_point + hws
    crop_start = np.maximum(start, 0)
    crop_end = np.minimum(end, shape)

    # Ensure each dimension is at least 1, mainly to support cases where the hws is 0 in one dimension
    crop_end = np.maximum(crop_end, crop_start + 1)

    # Crop image
    if len(shape) == 3 :
        cropped = image[
            crop_start[0]:crop_end[0],
            crop_start[1]:crop_end[1],
            crop_start[2]:crop_end[2],
        ]
    elif len(shape) == 2 :
        cropped = image[
            crop_start[0]:crop_end[0],
            crop_start[1]:crop_end[1],
        ]

    # Out of bound cropping allowed, pad the cropped image to respect the cropping region dimensions
    if allow_oob :
        pad_before = np.maximum(-start, 0)
        pad_after = np.maximum(end - shape, 0)
        pad_width = tuple((int(pb), int(pa)) for pb, pa in zip(pad_before, pad_after))
        return np.pad(cropped, pad_width, mode='constant', constant_values=0)
    else :
        return cropped
    

def generate_grid(shape, grid_size, segm_mask=None) :
    """Generate points in a grid.

    Args:
        shape (_type_): Image shape
        grid_size (_type_): Number of points per dimension
        segm_mask (_type_, optional): Segmentation mask. Defaults to None.

    Returns:
        _type_: The generated points
    """
    x_coord = np.linspace(0, shape[1]-1, min(shape[1]-1, grid_size), dtype=int)
    y_coord = np.linspace(0, shape[0]-1, min(shape[0]-1, grid_size), dtype=int)
    X, Y = np.meshgrid(x_coord, y_coord)
    grid_coords = np.stack([X.ravel(), Y.ravel()], axis=-1)
    if segm_mask is not None:
        inside_mask = segm_mask[grid_coords[:,1], grid_coords[:,0]] == 1
        return grid_coords[inside_mask]
    return grid_coords


def filter_image(image, median_kernel, gaussian_kernel_xy, gaussian_kernel_z) :
    """Pass an image through a median and a gaussian filter

    Args:
        image (_type_): The original image
        median_kernel (_type_): Kernel for the median filter. 0 for no filtering.
        gaussian_kernel (_type_): Kernel for the gaussian filter. 0 For no filtering.

    Returns:
        _type_: Filtered image.
    """
    if median_kernel :
        image = medfilt(image, median_kernel)
    if gaussian_kernel_xy :
        if gaussian_kernel_z :
            image = gaussian(image, [gaussian_kernel_z, gaussian_kernel_xy, gaussian_kernel_xy])
        else :
            image = gaussian(image, gaussian_kernel_xy)
    return image

def threshold_image(image, type='otsu') :
    """Thresholds an image

    Args:
        image (_type_): The original image
        type (str, optional): Threshold type. Can be 'otsu' or 'mean'. Defaults to 'otsu'.

    Returns:
        _type_: Binary image.
    """
    if type=='otsu' :
        threshold = threshold_otsu(image)
    if type =='mean' :
        threshold = threshold_mean(image) 
    thresolded = (image > threshold).astype(int)
    return thresolded

def sample_points_in_binary_mask(binary, num_points=20) :
    """Uniformly sample points in a binary shape.

    Args:
        binary (_type_): The binary image
        num_points (int, optional): The number of points to sample. Defaults to 20.

    Returns:
        _type_: _description_
    """
    coords = np.array(np.where(binary))
    indices = np.linspace(0, coords.shape[1]-1, num_points, dtype=int)
    sampled_points = coords[:,indices]
    return sampled_points.transpose(1,0)[:,::-1]

def filter_and_threshold(image, median_kernel=0, gaussian_kernel=11, threshold_type='otsu') :
    """Fitler an image with median and gaussian filtering, and thresholds it

    Args:
        image (_type_): _description_
        median_kernel (int, optional): _description_. Defaults to 0.
        gaussian_kernel (int, optional): _description_. Defaults to 11.
        threshold_type (str, optional): _description_. Defaults to 'otsu'.

    Returns:
        _type_: The binary mask
    """
    
    filtered = filter_image(image, median_kernel=median_kernel, gaussian_kernel_xy=gaussian_kernel, gaussian_kernel_z=None)
    if threshold_type == 'otsu' :
        threshold = threshold_otsu(filtered)
    elif threshold_type == 'mean' :
        threshold = threshold_mean(filtered)
    thresholded = (filtered > threshold).astype(int)
    return thresholded

def generate_uniform_grid_in_region(image, center_point, hws, grid_size, median_kernel=0, gaussian_kernel=11, threshold_type='otsu', return_mask=False) :
    """Generates points in a grid in a box.

    Args:
        image (_type_): The original image
        center_point (_type_): The center point of the box.
        hws (_type_): The half window size of the box.
        grid_size (_type_): The distance between points.
        median_kernel (int, optional): The kernel for median filtering, 0 for no filtering. Defaults to 0.
        gaussian_kernel (int, optional): The kernel for gaussian filtering, 0 for no filtering. Defaults to 11.
        threshold_type (str, optional): The thresold type, can be 'otsu' or 'mean'. Defaults to 'otsu'.

    Returns:
        _type_: The generated points
    """
    ndims = image.ndim
    if ndims > 2 :
        # Convert to grayscale
        image = image[...,0]
    segm_mask = get_box(image.shape, center_point, hws)
    thresholded = filter_and_threshold(image, median_kernel, gaussian_kernel, threshold_type)
    grid_points = generate_grid(image.shape, grid_size=grid_size, segm_mask=thresholded * segm_mask)
    if return_mask :
        return grid_points, thresholded
    return grid_points


def generate_uniform_grid_in_region_list(image, center_point_list, hws_list, grid_size, median_kernel=0, gaussian_kernel=11, threshold_type="otsu", return_mask=False) :
    ndims = image.ndim
    assert ndims == 2
    assert len(center_point_list) == len(hws_list)

    thresholded = filter_and_threshold(image, median_kernel, gaussian_kernel, threshold_type)
    grid_coords = generate_grid(image.shape, grid_size=grid_size)

    points = []
    points_lengths = []

    for center_point, hws in zip(center_point_list, hws_list) :
        box = get_box(image.shape, center_point, hws)
        segm_mask = np.logical_and(box, thresholded)
        inside_mask = segm_mask[grid_coords[:,1], grid_coords[:,0]]
        new_points = grid_coords[inside_mask]
        points.append(new_points)
        points_lengths.append(len(new_points))
    points = np.vstack(points)

    if return_mask :
        return points, points_lengths, thresholded
    return points, points_lengths




def compute_Fis(H, W, rois_list) :
    Fis = []
    for xmin, xmax, ymin, ymax in rois_list :
        sx_min = xmax - W
        sx_max = xmin
        sy_min = ymax - H
        sy_max = ymin
        # If the ROI is too big to be contained in the window, skip it
        if (sx_max < sx_min) or (sy_max < sy_min) :
            continue
        Fis.append((sx_min, sx_max, sy_min, sy_max))
    return Fis


#######################################
# Maximum overlapping intervals (1D), using a sweep line algorithm
#######################################

def sweepLine1D(arr) :
    # intervals are (ymin, ymax, weight)
    # Create event for each start and end of segments
    events = []
    for ymin, ymax, w in arr :
        events.append((ymin, +w))
        events.append((ymax, -w))

    # Sort events by coordinate, then by type (end is processed before start)
    events.sort(key=lambda e: (e[0], e[1]))

    active = 0
    max_count = 0
    intervals = []
    last_y = None

    for y, typ in events :
        if last_y is not None and active == max_count and y > last_y:
            intervals.append((last_y, y))
        active += typ
        if active > max_count:
            max_count = active
            intervals = []
            last_y = y
        elif active == max_count :
            last_y = y
        else :
            last_y = y

    return max_count, intervals


#######################################
# Maximum overlapping regions (2D), using a sweep line algorithm
# Finds the maximum number of overlaps region in O(N^2 log(N)) time
# A Segment tree instead of a sweep for the y coordinate can reduce the complexity to O(N log(N))
#######################################

def sweepLine2D(rectangles) :
    # rectangles are (xmin, xmax, ymin, ymax, weight)

    # Create event in x
    events = []
    for xmin, xmax, ymin, ymax, w in rectangles:
        events.append((xmin, +w, (xmin, xmax, ymin, ymax)))
        events.append((xmax, -w, (xmin, xmax, ymin, ymax)))

    # Sort by x coordinates, and by type (leaves are processed before enters)
    events.sort(key=lambda e: (e[0], e[1]))

    active = []
    max_count = 0
    best_region = None

    for i in range(len(events) - 1) :
        x, w, (xmin, xmax, ymin, ymax) = events[i]
        if w > 0: # start
            active.append((ymin, ymax, w))
        else : #end
            active.remove((ymin, ymax, -w))

        next_x = events[i + 1][0]
        if next_x == x :
            continue # Zero width slab, skip

        if active:
            count, y_intervals = sweepLine1D(active)
            if count > max_count :
                max_count = count
                # Take the first max interval in case of ties
                if y_intervals :
                    ymin, ymax = y_intervals[0]
                    best_region = (x, next_x, ymin, ymax)

    return max_count, best_region




def clamp(val, low, high) :
    if val < low :
        return low
    elif val > high :
        return high
    return val



###############################################
# Prioritized intersection
###############################################

def compute_intersection(rectangles) :
    xmin = max([rect[0] for rect in rectangles])
    xmax = min([rect[1] for rect in rectangles])
    ymin = max([rect[2] for rect in rectangles])
    ymax = min([rect[3] for rect in rectangles])

    return (xmin, xmax, ymin, ymax)


def prioritized_intersection(Fis) :
    # Compute the intersection of intersection(F until n-1) and F n
    # If the intersection does not exists, skip F n
    # F 1 will always be included in the final intersection
    # Weaker Fis are discarded over stronger ones

    current = [Fis[0]]
    selected = [0] # indices of kept ROIs
    final_intersection = Fis[0]

    for i, test_set in enumerate(Fis[1:], start=1) :
        new_set = current + [test_set]
        intersection = compute_intersection(new_set)
        if intersection[0] <= intersection[1] and intersection[2] <= intersection[3] : # If intersection exists (min < max)
            selected.append(i)
            current = [intersection]
            final_intersection = intersection
    
    return final_intersection, selected