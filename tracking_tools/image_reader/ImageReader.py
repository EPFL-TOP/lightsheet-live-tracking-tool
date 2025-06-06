import tifffile
from pathlib import Path
from ..logger.logger import init_logger

class ImageReader() :
    def __init__(self, dirpath, log) :
        self.log = log
        self.folder = Path(dirpath) # Timelapse folder
        # Set default logger
        self.logger = init_logger("ImageReader")

    def read_image(self, position_name, settings, channel, time_point) :
        image_path = self.folder / f"{position_name}_{settings}" / f"t{time_point:04}_{channel}.tif"
        if not image_path.exists():
            self.logger.error(f"Missing image at{image_path}")
            return None
        try :
            image = tifffile.imread(str(image_path))
            if self.log : 
                self.logger.info(f"Read image {image_path}")
                self.logger.info(f"Image shape : {image.shape}")
            return image
        except Exception as e:
            self.logger.error(f'Cannot read {image_path}: {e}')
            return None