"""Tests for the bounding box interceptor algorithm."""

import unittest
import numpy as np

from dt_model.model.bbox import Box, Point, Result, compute


class TestBBoxInterceptor(unittest.TestCase):
    """Tests for the bounding box interceptor algorithm."""

    def setUp(self):
        """Create common test box."""
        self.box = Box(Point(0, 0), Point(1000, 1000))

    def test_perfect_vertical_line(self):
        """Test regression for points forming perfect vertical line."""
        x_values = np.array([500.0] * 5)  # All points at x=500
        y_values = np.array([100.0, 200.0, 300.0, 400.0, 500.0])

        result = compute(x_values, y_values, self.box)
        self.assertIsNotNone(result)
        assert result is not None

        # Should give vertical line at x=500
        self.assertAlmostEqual(result.segment.start.x, 500.0)
        self.assertAlmostEqual(result.segment.end.x, 500.0)
        self.assertAlmostEqual(result.segment.start.y, 0.0)
        self.assertAlmostEqual(result.segment.end.y, 1000.0)

    def test_perfect_horizontal_line(self):
        """Test regression for points forming perfect horizontal line."""
        x_values = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        y_values = np.array([500.0] * 5)  # All points at y=500

        result = compute(x_values, y_values, self.box)
        self.assertIsNotNone(result)
        assert result is not None

        # Should give horizontal line at y=500
        self.assertAlmostEqual(result.segment.start.x, 0.0)
        self.assertAlmostEqual(result.segment.end.x, 1000.0)
        self.assertAlmostEqual(result.segment.start.y, 500.0)
        self.assertAlmostEqual(result.segment.end.y, 500.0)

    def test_diagonal_line(self):
        """Test regression for points forming diagonal line."""
        x_values = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        y_values = x_values  # y = x

        result = compute(x_values, y_values, self.box)
        self.assertIsNotNone(result)
        assert result is not None

        # Should give y = x line intersecting at (0,0) and (1000,1000)
        self.assertAlmostEqual(result.segment.start.x, 0.0)
        self.assertAlmostEqual(result.segment.start.y, 0.0)
        self.assertAlmostEqual(result.segment.end.x, 1000.0)
        self.assertAlmostEqual(result.segment.end.y, 1000.0)

    def test_empty_input(self):
        """Test behavior with empty input arrays."""
        result = compute(np.array([]), np.array([]), self.box)
        self.assertIsNone(result)

    def test_single_point(self):
        """Test behavior with single point."""
        result = compute(np.array([500.0]), np.array([500.0]), self.box)
        self.assertIsNone(result)

    def test_steep_line(self):
        """Test near-vertical but not perfect vertical line."""
        x_values = np.array([495.0, 497.0, 499.0, 501.0, 503.0])
        y_values = np.array([100.0, 300.0, 500.0, 700.0, 900.0])

        result = compute(x_values, y_values, self.box)
        self.assertIsNotNone(result)

        # Line should be nearly vertical around x=500

    def test_shallow_line(self):
        """Test near-horizontal but not perfect horizontal line."""
        x_values = np.array([100.0, 300.0, 500.0, 700.0, 900.0])
        y_values = np.array([495.0, 497.0, 499.0, 501.0, 503.0])

        result = compute(x_values, y_values, self.box)
        self.assertIsNotNone(result)

        # Line should be nearly horizontal around y=500

    def test_line_outside_box(self):
        """Test line that doesn't intersect box."""
        x_values = np.array([1500.0, 1600.0, 1700.0, 1800.0, 1900.0])
        y_values = np.array([1500.0, 1600.0, 1700.0, 1800.0, 1900.0])

        # This should return None since line is completely outside box
        result = compute(x_values, y_values, self.box)
        self.assertIsNone(result)

    def test_line_partially_outside_box(self):
        """Test line that partially intersects box."""
        x_values = np.array([500.0, 1000.0, 1500.0, 2000.0])
        y_values = np.array([500.0, 1000.0, 1500.0, 2000.0])

        result = compute(x_values, y_values, self.box)
        self.assertIsNotNone(result)
        assert result is not None

        # Check that returned segment is within box bounds
        self.assertTrue(0 <= result.segment.start.x <= 1000)
        self.assertTrue(0 <= result.segment.start.y <= 1000)
        self.assertTrue(0 <= result.segment.end.x <= 1000)
        self.assertTrue(0 <= result.segment.end.y <= 1000)


if __name__ == "__main__":
    unittest.main()
