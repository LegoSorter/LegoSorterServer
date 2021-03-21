import logging

from collections import OrderedDict
from typing import List


class SimpleOrdering:
    def __init__(self):
        self.memorized_state: OrderedDict = OrderedDict()
        self.processed_bricks = []
        self.head_index = -1  # this indicates the index of the first brick on the tape

    def process_current_results(self, results):
        if len(results) == 0:
            logging.info("[SimpleOrdering] No bricks detected. It means that all bricks have surpassed the camera line")
            self._extract_processed_bricks(len(self.memorized_state))
            return

        first_brick_from_history = self._get_first_brick()

        if len(first_brick_from_history) == 0:
            logging.info(f"[SimpleOrdering] Nothing in history, adding all results and moving the head index by 1")

            self.head_index = self.head_index + 1
            self._add_results_to_current_state(results, start_from=self.head_index)
            return

        first_brick_from_results = results[0]

        if self._is_the_same_brick(first_brick_from_history, first_brick_from_results):
            logging.info(f"[SimpleOrdering] No brick has surpassed the camera line."
                         f"\n\t\t\t First brick from the history:"
                         f"\n\t\t\t {first_brick_from_history}"
                         f"\n\t\t\t Is the same brick as the current first brick:"
                         f"\n\t\t\t {first_brick_from_results}")
            self._add_results_to_current_state(results, start_from=self.head_index)
            return
        else:
            logging.info(f"[SimpleOrdering] Another brick detected at the head position. "
                         f"It means that the previous first brick has surpassed the camera line.")
            passed_bricks_count = self._get_count_of_passed_bricks(current_state=results)
            if passed_bricks_count >= 2:
                logging.error(
                    f"[SimpleOrdering] {passed_bricks_count} bricks have overpassed the camera line!"
                    f"Such a state shouldn't happen. Sorting results can be incorrect.")

            self._extract_processed_bricks(count=passed_bricks_count)
            self._add_results_to_current_state(results, start_from=self.head_index)
        pass

    def _add_results_to_current_state(self, results, start_from: int):
        for index, result in enumerate(results):
            history_of_brick = self.memorized_state.get(start_from + index, [])
            history_of_brick.append(result)
            self.memorized_state[start_from + index] = history_of_brick

        logging.info(f"[SimpleOrdering] Added results, the current state is:"
                     f"\n {list(self.memorized_state.items())}")

    def get_current_state(self):
        """
        Returns the memorized state of the belt in the following form:
            { index_of_the_brick: ((bounding_box), label, score), ... }
        """
        current_state = dict()
        for key, value in self.memorized_state.items():
            current_state[key] = value[-1]  # assign the most recent value

        return current_state

    def _get_first_brick(self):
        return self.memorized_state.get(self.head_index, [()])[-1]

    def reset(self):
        self.memorized_state.clear()
        self.head_index = -1
        self.processed_bricks = []

    def _is_the_same_brick(self, older_view, current_view):
        # Check if the position of the bounding box moved along the tape direction
        return older_view[0][0] <= current_view[0][0]

    def _extract_processed_bricks(self, count):
        for i in range(count):
            first = self.memorized_state.pop(self.head_index + i)
            self.processed_bricks.append(first)
            logging.info(f"[SimpleOrdering] A brick moved to the processed queue:\n {first}")

        self.head_index = self.head_index + count

    def get_count_of_results_to_send(self) -> int:
        return len(self.processed_bricks)

    def pop_first_processed_brick(self) -> List:
        if len(self.processed_bricks) == 0:
            return []

        return self.processed_bricks.pop(0)

    def _get_count_of_passed_bricks(self, current_state) -> int:
        first_brick_on_the_tape_current = current_state[0]

        passed_count = 0
        for brick_snapshots in self.memorized_state.values():
            last_snapshot = brick_snapshots[-1]

            if first_brick_on_the_tape_current[0][0] <= last_snapshot[0][0]:
                passed_count = passed_count + 1
            else:
                break

        return passed_count
