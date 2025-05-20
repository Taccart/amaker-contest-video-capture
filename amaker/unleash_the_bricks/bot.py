import logging
from enum import Enum
from typing import List, Tuple, Optional

import numpy as np


DEFAULT_COLOR = (0, 0, 250)
DEFAULT_TRAIL_LENGTH = 10


class BotStatus(Enum):
    """Enum for bot states"""
    UNKNOWN = -1
    WAITING = 0
    MOVING = 1
    SEARCHING = 2
    FETCHING = 3
    CATCHING = 4
    DROPPING = 5
    STOPPED = 6
    SAFETY = 10
    COMPLETED = 20
    PANIC = 100


class UnleashTheBrickBot:
    """Class to track a bot's position and color (for identification and video feedback)"""

    def __init__(self, bot_id: int, name: str = None,
                 rgb_color=DEFAULT_COLOR
                 , trail_length: int = DEFAULT_TRAIL_LENGTH):
        self.name = name
        self.id = bot_id
        self.color = rgb_color
        self.trail_length = trail_length
        self.status_history = []
        self.tag_position_history = []
        self.trail=[]
        self.status: BotStatus = BotStatus.UNKNOWN
        self.total_distance = 0
        self.collected_count = 0

    def add_collected(self, amount: int = 1):
        self.collected_count += amount


    def add_tag_position(self, tag):
        """Add a new position to the bot's trail"""
        previous_pos = self.get_last_tag_position()

        if previous_pos is not None:
            # Only add position if it's different from the previous one
            current_coords=tag.corners[0].astype(float)
            previous_coords=previous_pos.corners[0].astype(float)
            if not np.array_equal(current_coords, previous_coords):
                distance = np.linalg.norm(np.array(current_coords) - np.array(previous_coords))
                self.total_distance += distance
                #only add a position when it's changed
                self.tag_position_history.append(tag)
        else:
            # If no previous position, just add the current one
            self.tag_position_history.append(tag)
        while len(self.tag_position_history) > self.trail_length:
            self.tag_position_history.pop(0)


    def get_last_tag_position(self) -> Optional[Tuple[float, float, float]]:
        """Get the last known position of the bot"""
        if self.tag_position_history:
            return self.tag_position_history[-1]
        return None


    def set_bot_status(self, state: BotStatus):
        """Set the bot's state"""
        self.status = state
        logging.info(f"Bot {self.name}:{self.id} state changed to {self.status.name}")



    def get_trail(self) -> List:
        """Get the bot's trail"""
        return self.tag_position_history

    def get_bot_info(self) -> str:
        """Get bot information"""
        return (
            f"{self.name:>20} [{self.status.name:<10}] {self.collected_count:>3}/{(self.total_distance/100):4.2f}")

    def __repr__(self):
        return f"BotTracker(name={self.name}, id={self.id}, status={self.status}"
