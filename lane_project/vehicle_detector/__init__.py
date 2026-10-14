"""Classical (no model, no weights) vehicle detection that sits on top of the lane detector."""
from .config import VConfig
from .detector import VehicleDetector, Vehicle
from .overlay import draw_vehicles

__all__ = ["VConfig", "VehicleDetector", "Vehicle", "draw_vehicles"]
