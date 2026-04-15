class GateKeeperLogic:
    def __init__(self, line_y, buffer_threshold=5):
        self.line_y = line_y
        self.buffer_threshold = buffer_threshold
        # Stores {track_id: {"side": -1/1, "frames": count}}
        self.track_states = {}
        self.counts = {"In": 0, "Out": 0}

    def check_crossing(self, track_id, current_y):
        # -1 = Above the line, 1 = Below the line
        current_side = -1 if current_y < self.line_y else 1
        
        if track_id not in self.track_states:
            self.track_states[track_id] = {"side": current_side, "frames": 1}
            return None

        state = self.track_states[track_id]

        if current_side == state["side"]:
            # If they stay on the same side, increment their "confidence"
            state["frames"] = min(state["frames"] + 1, self.buffer_threshold + 1)
        else:
            # If they switched sides, check if they were on the previous side long enough
            if state["frames"] >= self.buffer_threshold:
                direction = "In" if current_side == 1 else "Out"
                self.counts[direction] += 1
                # Reset state to the new side
                self.track_states[track_id] = {"side": current_side, "frames": 1}
                return direction
            else:
                # This was likely a "jitter" or a flicker. Just update side without counting.
                self.track_states[track_id] = {"side": current_side, "frames": 1}
        
        return None