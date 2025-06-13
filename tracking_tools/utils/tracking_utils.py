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
        _type_: Thresholded binary image.
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


def generate_uniform_grid_in_region(image, center_point, hws, grid_size, median_kernel=0, gaussian_kernel=11, threshold_type='otsu') :
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
    filtered = filter_image(image, median_kernel=median_kernel, gaussian_kernel_xy=gaussian_kernel, gaussian_kernel_z=None)
    if threshold_type == 'otsu' :
        threshold = threshold_otsu(filtered)
    elif threshold_type == 'mean' :
        threshold = threshold_mean(filtered)
    thresholded = (filtered > threshold).astype(int)
    grid_points = generate_grid(image.shape, grid_size=grid_size, segm_mask=thresholded * segm_mask)
    return grid_points