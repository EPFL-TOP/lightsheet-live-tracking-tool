from dataclasses import dataclass
import numpy as np
from typing import Union, Tuple, List

@dataclass
class Position2D:
    x:float
    y:float

    def to_array(self, order: str = "yx") -> np.ndarray:
        valid_axes = {'x': self.x, 'y': self.y}
        try:
            return np.array([valid_axes[axis] for axis in order])
        except KeyError as e:
            raise ValueError(f"Invalid axis '{e.args[0]}' in order string. Use combination of 'x', 'y'.")
        
    def to_position3D(self, z) -> 'Position3D' :
        return Position3D(x=self.x, y=self.y, z=z)
        
    def __add__(self, shift: Union['Position2D', 'Shift2D']) -> 'Position2D':
        if isinstance(shift, (Position2D, Shift2D)):
            return Position2D(x=self.x + shift.x, y=self.y + shift.y)
        else:
            raise TypeError("Shift must be Position2D, or Shift2D.")
        
    def __mul__(self, scalar: Union[int, float]) -> 'Position2D' :
        return Position2D(x=self.x * scalar, y=self.y * scalar)
    
    def __rmul__(self, scalar: Union[int, float]) -> 'Position2D' :
        return self.__mul__(scalar)
    
    @classmethod
    def invalid(cls) :
        return cls(x=np.nan, y=np.nan)


@dataclass
class Shift2D:
    x:float
    y:float



@dataclass
class Position3D:
    x:float
    y:float
    z:float

    def to_array(self, order: str = "zyx") -> np.ndarray:
        valid_axes = {'x': self.x, 'y': self.y, 'z': self.z}
        try:
            return np.array([valid_axes[axis] for axis in order])
        except KeyError as e:
            raise ValueError(f"Invalid axis '{e.args[0]}' in order string. Use combination of 'x', 'y', 'z'.")
        
    def to_position2D(self) -> Position2D :
        return Position2D(x=self.x, y=self.y)
    
    def __add__(self, shift: Union['Position3D', 'Shift3D']) -> 'Position3D':
        if isinstance(shift, (Position3D, Shift3D)):
            return Position3D(x=self.x + shift.x, y=self.y + shift.y, z=self.z + shift.z)
        else:
            raise TypeError("Shift must be Position3D, or Shift3D")
        
    def __sub__(self, shift: Union['Position3D', 'Shift3D']) -> 'Position3D':
        if isinstance(shift, (Position3D, Shift3D)):
            return Position3D(x=self.x - shift.x, y=self.y - shift.y, z=self.z - shift.z)
        else:
            raise TypeError("Shift must be Position3D, or Shift3D")
        
    @classmethod
    def invalid(cls) :
        return cls(x=np.nan, y=np.nan, z=np.nan)


@dataclass
class Shift3D:
    x:float
    y:float
    z:float

    @classmethod
    def from_positions(cls, pos1: 'Position3D', pos2: 'Position3D') -> 'Shift3D':
        """Shift3D class constructor from 2 Position3D (pos1 - pos2)

        Args:
            pos1 (Position3D): 
            pos2 (Position3D):

        Returns:
            Shift3D:
        """
        return cls(
            x=pos1.x - pos2.x,
            y=pos1.y - pos2.y,
            z=pos1.z - pos2.z
        )


@dataclass
class ROI:
    x: float
    y: float
    width: float
    height: float
    order : int

    def to_position2D(self) -> Position2D :
        return Position2D(x=self.x, y=self.y)
    
    def to_position3D(self, z) -> Position3D :
        return Position3D(x=self.x, y=self.y, z=z)
    
    def __mul__(self, scalar: Union[int, float]) -> 'ROI' :
        return ROI(x=self.x * scalar, y=self.y * scalar, height=self.height * scalar, width=self.width * scalar, order=self.order)
    
    def __rmul__(self, scalar: Union[int, float]) -> 'ROI' :
        return self.__mul__(scalar)
    
    @classmethod
    def invalid(cls, order) :
        return cls(x=np.nan, y=np.nan, height=np.nan, width=np.nan, order=order)