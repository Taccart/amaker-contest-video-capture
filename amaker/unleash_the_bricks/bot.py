import logging
from enum import Enum
from typing import List, Tuple, Optional
from collections import deque
import numpy as np


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
    DROPPING = 5
    STOPPED = 6
    SAFETY = 10
    COMPLETED = 20
    PANIC = 100


class UnleashTheBrickBot:
    """Class to track a bot's position and color (for identification and video feedback)"""

    def __init__(self, tag_id: int, name: str = None,
                 trail_color=DEFAULT_TRAIL_COLOR, trail_length: int = DEFAULT_TRAIL_LENGTH):
        self.name = name
        self.id = tag_id
        self.trail_color = trail_color
        self.trail_length = trail_length
        self.status_history = []
        self.position_history = deque(maxlen=trail_length)
        self.status: BotStatus = BotStatus.UNKNOWN
        self.total_distance = 0
        self.collected_count = 0

    def add_collected(self, amount: int = 1):
        self.collected_count += amount

    def get_collected(self):
        return self.collected_count


    def add_position(self, position):
        """Add a new position to the bot's trail"""
        previous_pos = self.get_last_position()

        if previous_pos is not None:
            # Only add position if it's different from the previous one
            if not np.array_equal(position, previous_pos):
                distance = np.linalg.norm(np.array(previous_pos) - np.array(position))
                self.total_distance += distance
                self.position_history.append(position)
        else:
            # First position
            self.position_history.append(position)


    def get_last_position(self) -> Optional[Tuple[float, float, float]]:
        """Get the last known position of the bot"""
        if self.position_history:
            return self.position_history[-1]
        return None


    def set_bot_status(self, state: BotStatus):
        """Set the bot's state"""
        self.status = state
        logging.info(f"Bot {self.name}:{self.id} state changed to {self.status.name}")

    def get_bot_status(self) -> BotStatus:
        """Get the current state of the bot"""
        return self.status


    def get_total_distance(self) -> float:
        """Get the total distance traveled by the bot"""
        return self.total_distance

    def get_trail(self) -> List:
        """Get the bot's trail"""
        return self.position_history

    def get_bot_info(self) -> str:
        """Get bot information"""
        return (
            f"Bot {self.id:>3}:{self.name:<15}. Status: {self.get_bot_status().name:<8}. Declared collected : {self.get_collected():<3}. Total distance: {self.get_total_distance():.2f}")
    def _resize_logo(self, target_width):
        if not self.logo_loaded or self.logo_image is None:
            return

        logo_scale_factor = target_width / DEFAULT_SCREEN_WIDTH
        if logo_scale_factor != 1.0:
            logo_width = int(self.logo_image.shape[1] * logo_scale_factor)
            logo_height = int(self.logo_image.shape[0] * logo_scale_factor)
            self.scaled_logo = cv2.resize(self.logo_image, (logo_width, logo_height))
        else:
            self.scaled_logo = self.logo_image
    def __repr__(self):
        return f"BotTracker(name={self.name}, id={self.id}, status={self.status}"
