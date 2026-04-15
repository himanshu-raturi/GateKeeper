"""tests/test_logic.py — unit tests for src/logic.py.

Covers:
  • get_base_point
  • cross_product
  • segments_intersect
  • side_of_line
  • DirectionalBuffer
  • VelocityEstimator
  • TripwireCounter (crossing events, in/out counts, net count, cleanup)
"""

import time
import unittest
from unittest.mock import patch

from src.logic import (
    DirectionalBuffer,
    TripwireCounter,
    VelocityEstimator,
    cross_product,
    get_base_point,
    segments_intersect,
    side_of_line,
)


# ---------------------------------------------------------------------------
# get_base_point
# ---------------------------------------------------------------------------

class TestGetBasePoint(unittest.TestCase):
    def test_midpoint_x(self):
        bp = get_base_point((100, 50, 200, 300))
        self.assertAlmostEqual(bp[0], 150.0)

    def test_bottom_y(self):
        bp = get_base_point((100, 50, 200, 300))
        self.assertAlmostEqual(bp[1], 300.0)

    def test_square_box(self):
        bp = get_base_point((0, 0, 10, 10))
        self.assertEqual(bp, (5.0, 10.0))

    def test_non_integer_coords(self):
        bp = get_base_point((10.5, 20.3, 30.5, 40.7))
        self.assertAlmostEqual(bp[0], 20.5)
        self.assertAlmostEqual(bp[1], 40.7)


# ---------------------------------------------------------------------------
# cross_product
# ---------------------------------------------------------------------------

class TestCrossProduct(unittest.TestCase):
    def test_positive_left(self):
        # B is to the left of O→A (going right)
        cp = cross_product((0, 0), (1, 0), (0, 1))
        self.assertGreater(cp, 0)

    def test_negative_right(self):
        # B is to the right of O→A
        cp = cross_product((0, 0), (1, 0), (0, -1))
        self.assertLess(cp, 0)

    def test_collinear(self):
        cp = cross_product((0, 0), (1, 0), (2, 0))
        self.assertEqual(cp, 0)

    def test_known_value(self):
        # (0,0)→(3,0) cross with (0,2): 3*2 - 0*0 = 6
        cp = cross_product((0, 0), (3, 0), (0, 2))
        self.assertAlmostEqual(cp, 6.0)


# ---------------------------------------------------------------------------
# segments_intersect
# ---------------------------------------------------------------------------

class TestSegmentsIntersect(unittest.TestCase):
    def test_clear_cross(self):
        # Vertical segment crosses horizontal line
        self.assertTrue(
            segments_intersect((5, -5), (5, 5), (0, 0), (10, 0))
        )

    def test_no_cross_parallel(self):
        self.assertFalse(
            segments_intersect((0, 1), (10, 1), (0, 0), (10, 0))
        )

    def test_no_cross_same_side(self):
        # Both endpoints of P on the same side of Q
        self.assertFalse(
            segments_intersect((0, 5), (10, 5), (0, 0), (10, 0))
        )

    def test_diagonal_cross(self):
        # Diagonal segments that clearly intersect
        self.assertTrue(
            segments_intersect((0, 0), (4, 4), (0, 4), (4, 0))
        )

    def test_collinear_no_cross(self):
        # Collinear points — treated as non-crossing per spec
        self.assertFalse(
            segments_intersect((0, 0), (5, 0), (3, 0), (10, 0))
        )


# ---------------------------------------------------------------------------
# side_of_line
# ---------------------------------------------------------------------------

class TestSideOfLine(unittest.TestCase):
    def test_left_side(self):
        # Point at y=-1 is above line y=0 in screen coords.
        # cross_product((0,0),(10,0),(5,-1)) = 10*(-1) - 0*5 = -10 → side -1
        self.assertEqual(side_of_line((5, -1), (0, 0), (10, 0)), -1)

    def test_right_side(self):
        # Point at y=1 is below line y=0 in screen coords → side +1
        self.assertEqual(side_of_line((5, 1), (0, 0), (10, 0)), 1)

    def test_on_line(self):
        self.assertEqual(side_of_line((5, 0), (0, 0), (10, 0)), 0)


# ---------------------------------------------------------------------------
# DirectionalBuffer
# ---------------------------------------------------------------------------

class TestDirectionalBuffer(unittest.TestCase):
    def setUp(self):
        self.buf = DirectionalBuffer(required_frames=3)

    def test_not_confirmed_below_threshold(self):
        for _ in range(2):
            result = self.buf.update(1, 1)
        self.assertIsNone(result)

    def test_confirmed_at_threshold(self):
        for _ in range(3):
            result = self.buf.update(1, 1)
        self.assertEqual(result, 1)

    def test_reset_on_side_change(self):
        for _ in range(3):
            self.buf.update(1, 1)
        # Switch side — should restart streak
        for _ in range(2):
            result = self.buf.update(1, -1)
        # Still seeing old confirmed side until new streak completes
        self.assertEqual(result, 1)

    def test_collinear_does_not_reset(self):
        for _ in range(2):
            self.buf.update(1, 1)
        self.buf.update(1, 0)   # collinear — should be ignored
        result = self.buf.update(1, 1)  # 3rd real +1
        self.assertEqual(result, 1)

    def test_multiple_tracks_independent(self):
        for _ in range(3):
            self.buf.update(1, 1)
            self.buf.update(2, -1)
        self.assertEqual(self.buf.get_confirmed(1),  1)
        self.assertEqual(self.buf.get_confirmed(2), -1)

    def test_remove_clears_state(self):
        for _ in range(3):
            self.buf.update(1, 1)
        self.buf.remove(1)
        self.assertIsNone(self.buf.get_confirmed(1))


# ---------------------------------------------------------------------------
# VelocityEstimator
# ---------------------------------------------------------------------------

class TestVelocityEstimator(unittest.TestCase):
    def test_no_speed_single_observation(self):
        ve = VelocityEstimator()
        ve.update(1, (0.0, 0.0))
        self.assertIsNone(ve.get_speed(1))

    def test_no_speed_unknown_track(self):
        ve = VelocityEstimator()
        self.assertIsNone(ve.get_speed(99))

    def test_positive_speed(self):
        ve = VelocityEstimator()
        t0 = time.monotonic()
        with patch("src.logic.time.monotonic", side_effect=[t0, t0 + 1.0]):
            ve.update(1, (0.0, 0.0))
            ve.update(1, (100.0, 0.0))
        speed = ve.get_speed(1)
        self.assertIsNotNone(speed)
        self.assertAlmostEqual(speed, 100.0, delta=1.0)

    def test_remove_clears_history(self):
        ve = VelocityEstimator()
        t0 = time.monotonic()
        with patch("src.logic.time.monotonic", side_effect=[t0, t0 + 1.0]):
            ve.update(1, (0.0, 0.0))
            ve.update(1, (50.0, 0.0))
        ve.remove(1)
        self.assertIsNone(ve.get_speed(1))


# ---------------------------------------------------------------------------
# TripwireCounter — integration-level tests
# ---------------------------------------------------------------------------

class TestTripwireCounter(unittest.TestCase):
    """The tripwire is a horizontal line at y=100 from x=0 to x=200.

    cross_product((0,100),(200,100),(x, y_test)):
      = (200-0)*(y_test-100) - (100-100)*(x-0) = 200*(y_test-100)
    So y < 100 (above line in screen) → side -1; y > 100 (below line) → side +1.

    Convention: side -1 → side +1 (crossing downward / into area) = "in"
                side +1 → side -1 (crossing upward / out of area)  = "out"
    """

    LINE_START = (0, 100)
    LINE_END   = (200, 100)

    def _make_counter(self, buf: int = 1) -> TripwireCounter:
        return TripwireCounter(
            line_start=self.LINE_START,
            line_end=self.LINE_END,
            buffer_in=buf,
            buffer_out=buf,
            velocity_enabled=False,
        )

    def _bbox_at_y(self, y: float):
        """Thin bounding box centred at x=100 with base point at y."""
        return (90.0, y - 10, 110.0, y)

    def test_no_event_same_side(self):
        counter = self._make_counter(buf=1)
        counter.update(1, self._bbox_at_y(50))   # side -1 (above line)
        event = counter.update(1, self._bbox_at_y(60))  # still side -1
        self.assertIsNone(event)
        self.assertEqual(counter.count_in, 0)
        self.assertEqual(counter.count_out, 0)

    def test_crossing_in(self):
        """Moving from above the line (side -1) to below (side +1) = 'in'."""
        counter = self._make_counter(buf=1)
        counter.update(1, self._bbox_at_y(50))    # side -1  (above line)
        event = counter.update(1, self._bbox_at_y(150))  # side +1  (below line)
        self.assertEqual(event, "in")
        self.assertEqual(counter.count_in, 1)
        self.assertEqual(counter.count_out, 0)

    def test_crossing_out(self):
        """Moving from below (side +1) to above (side -1) = 'out'."""
        counter = self._make_counter(buf=1)
        counter.update(1, self._bbox_at_y(150))   # side +1  (below line)
        event = counter.update(1, self._bbox_at_y(50))   # side -1  (above line)
        self.assertEqual(event, "out")
        self.assertEqual(counter.count_out, 1)
        self.assertEqual(counter.count_in, 0)

    def test_net_count(self):
        counter = self._make_counter(buf=1)
        # Two people enter, one leaves
        counter.update(1, self._bbox_at_y(50))
        counter.update(1, self._bbox_at_y(150))   # in
        counter.update(2, self._bbox_at_y(50))
        counter.update(2, self._bbox_at_y(150))   # in
        counter.update(3, self._bbox_at_y(150))
        counter.update(3, self._bbox_at_y(50))    # out
        self.assertEqual(counter.net_count, 1)

    def test_buffer_prevents_flicker(self):
        """With buf=3 a single frame on the new side should NOT trigger a count."""
        counter = self._make_counter(buf=3)
        # Confirm track 1 on side -1 for 3 frames
        for _ in range(3):
            counter.update(1, self._bbox_at_y(50))
        # One frame on the opposite side — not yet confirmed
        event = counter.update(1, self._bbox_at_y(150))
        self.assertIsNone(event)
        self.assertEqual(counter.count_in, 0)

    def test_buffer_confirms_after_n_frames(self):
        """After exactly buf frames on the new side, the crossing IS counted."""
        counter = self._make_counter(buf=3)
        for _ in range(3):
            counter.update(1, self._bbox_at_y(50))   # confirm side -1
        event = None
        for _ in range(3):
            event = counter.update(1, self._bbox_at_y(150))  # accumulate +1 side
        self.assertEqual(event, "in")
        self.assertEqual(counter.count_in, 1)

    def test_remove_track_clears_state(self):
        counter = self._make_counter(buf=1)
        counter.update(1, self._bbox_at_y(50))
        counter.remove_track(1)
        # After removal, moving to the other side should NOT count
        # because there is no prior confirmed side
        event = counter.update(1, self._bbox_at_y(150))
        self.assertIsNone(event)

    def test_multiple_independent_tracks(self):
        counter = self._make_counter(buf=1)
        counter.update(10, self._bbox_at_y(50))
        counter.update(20, self._bbox_at_y(150))
        event10 = counter.update(10, self._bbox_at_y(150))  # crossing in
        event20 = counter.update(20, self._bbox_at_y(50))   # crossing out
        self.assertEqual(event10, "in")
        self.assertEqual(event20, "out")
        self.assertEqual(counter.net_count, 0)

    def test_velocity_enabled_returns_speed(self):
        counter = TripwireCounter(
            line_start=self.LINE_START,
            line_end=self.LINE_END,
            velocity_enabled=True,
        )
        t0 = time.monotonic()
        with patch("src.logic.time.monotonic", side_effect=[t0, t0 + 1.0]):
            counter.update(1, self._bbox_at_y(50))
            counter.update(1, self._bbox_at_y(150))
        speed = counter.get_speed(1)
        self.assertIsNotNone(speed)
        self.assertGreater(speed, 0)

    def test_velocity_disabled_returns_none(self):
        counter = TripwireCounter(
            line_start=self.LINE_START,
            line_end=self.LINE_END,
            velocity_enabled=False,
        )
        counter.update(1, self._bbox_at_y(50))
        self.assertIsNone(counter.get_speed(1))


if __name__ == "__main__":
    unittest.main()
