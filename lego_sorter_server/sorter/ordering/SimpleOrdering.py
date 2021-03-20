import logging

from collections import OrderedDict


class SimpleOrdering:
    def __init__(self):
        self.current_state: OrderedDict = OrderedDict()
        self.processed_bricks = []
        self.head_index = 0

    def process_current_results(self, results):
        first_brick_from_history = self.get_first_brick()

        if len(first_brick_from_history) == 0:
            logging.info(f"[SimpleOrdering] Nothing in history, adding all results...")

            self._add_results_to_current_state(results, start_from=self.head_index)
            return

        first_brick_from_results = results[0]

        if self.is_the_same_brick(first_brick_from_history, first_brick_from_results):
            logging.info(f"[SimpleOrdering] No brick has surpassed the camera line."
                         f"\n\t\t\t First brick from the history:"
                         f"\n {first_brick_from_history}"
                         f"\n\t\t\t Is the same brick as the current first brick:"
                         f"\n {first_brick_from_results}")
            self._add_results_to_current_state(results, start_from=self.head_index)
            return
        else:
            logging.info(f"[SimpleOrdering] Another brick detected at the head position. "
                         f"It means that the previous first brick has surpassed the camera line.")
            missing_bricks_count = len(self.current_state) - len(results)
            if len(self.current_state) - len(results) >= 2:
                logging.error(
                    f"[SimpleOrdering] {missing_bricks_count} bricks are missing! Such a state shouldn't happen."
                    f"Sorting results can be incorrect. Consider resetting the sorting process")

            self._add_results_to_current_state(results, start_from=self.head_index + missing_bricks_count)
            self.extract_processed_bricks(count=missing_bricks_count)
        pass

    def _add_results_to_current_state(self, results, start_from: int):
        for index, result in enumerate(results):
            history_of_brick = self.current_state.get(start_from + index, [])
            history_of_brick.append(result)
            self.current_state[start_from + index] = history_of_brick

        logging.info(f"[SimpleOrdering] Added results, current state is:"
                     f"\n {list(self.current_state.items())}")
        self.head_index = self.head_index + len(results)

    def get_current_state(self):
        return len(self.current_state.items())

    def get_first_brick(self):
        return next(iter(self.current_state.items()), [])

    def reset(self):
        self.current_state.clear()

    def is_the_same_brick(self, older_view, current_view):
        # Check if the position of the bounding box moved along the tape direction
        return older_view[0][0] >= current_view[0][0]

    def extract_processed_bricks(self, count):
        for i in range(count):
            first = self.current_state.pop(self.head_index + i)
            self.processed_bricks.append(first)
            logging.info(f"[SimpleOrdering] Brick moved to processed queue:\n {first}")

    def pop_first_processed_brick(self):
        if len(self.processed_bricks) == 0:
            return []

        return self.processed_bricks.pop(0)
