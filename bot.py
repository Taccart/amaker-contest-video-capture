import logging
from typing import List
from enum import Enum
import numpy as np

# Bot tracker constants
DEFAULT_BOT_COLOR_A = (0, 0, 250)
DEFAULT_BOT_COLOR_B = (0, 0, 255)
DEFAULT_TRAIL_COLOR = (0, 0, 250)
DEFAULT_TRAIL_LENGTH = 20

class BotStatus(Enum):
    """Enum for bot states"""
    UNKNOWN = -1
    WAITING = 0
    MOVING = 1
    SEARCHING = 2
    FETCHING = 3
    CATCHING = 4
    DROPING = 5
    STOPPED = 6
    TO_SAFETY = 10
    MISSON_COMPLETED = 20

class Bot:
    """Class to track a bot's position and color (for identification and video feedback)"""

    def __init__(self, name: str = "microbot", id: int = None, color_a=DEFAULT_BOT_COLOR_A, color_b=DEFAULT_BOT_COLOR_B,
                 trail_color=DEFAULT_TRAIL_COLOR, trail_length: int = DEFAULT_TRAIL_LENGTH):
        self.name = name
        self.id = id
        self.color_a = color_a
        self.color_b = color_b
        self.trail_color = trail_color
        self.trail_length = trail_length
        self.trail = []
        self.status: BotStatus = BotStatus.UNKNOWN
        self.total_distance = 0

    def add_position(self, position):
        """Add a new position to the bot's trail"""
        self.trail.append(position)
        if len(self.trail) > self.trail_length:
            self.trail.pop(0)
        self.total_distance += self.calculate_distance(position)
        logging.debug(f"bot {self.name}:{self.id}, position: {self.get_last_position()}, total distance: {self.total_distance:.2f}")

    def get_last_position(self) -> tuple|None :
        """Get the last known position of the bot"""
        if self.trail:
            return self.trail[-1]
        else:
            return None

    def set_bot_status(self, state: BotStatus):
        """Set the bot's state"""
        self.status = state
        logging.info(f"Bot {self.name}:{self.id} state changed to {self.status.name}")

    def get_bot_status(self) -> BotStatus:
        """Get the current state of the bot"""
        return self.status

    def calculate_distance(self, position) -> float:
        """Calculate the distance from the last position to the current position"""
        if len(self.trail) < 2:
            return 0
        last_position = self.trail[-2]
        distance = np.linalg.norm(np.array(position) - np.array(last_position))
        return distance

    def get_total_distance(self) -> float:
        """Get the total distance traveled by the bot"""
        return self.total_distance

    def get_trail(self) -> List:
        """Get the bot's trail"""
        return self.trail

    def get_bot_info(self) -> str:
        """Get bot information"""
        return f"{self.name}.{self.id}  is {self.get_bot_status()}"

    def __repr__(self):
        return f"BotTracker(name={self.name}, color_a={self.color_a}, color_b={self.color_b}, trail_color={self.trail_color}, trail_length={self.trail_length})"
