"""
Bounding box interceptor algorithm
==================================

This module implements an algorithm to find where a regression line
intersects a bounding box.

Given:
    - A cloud of points (x[i], y[i])
    - A bounding box with corners (min_x, min_y) and (max_x, max_y)

The algorithm:
    1. Computes best-fit line through points using linear regression
    2. Finds where this line intersects the bounding box edges
    3. Returns these intersection points as a line segment

Example box:
 (0; 1,000) +-----------------+ (1,000; 1,000)
            |                 |
            |                 |
            |                 |
            |                 |
            |                 |
      (0;0) +-----------------+ (1,000, 0)

Used by the overtourism dashboard to draw median lines in phase space diagrams.
"""

# TODO(bassosimone):
# - Handle scipy warnings in stats.linregress for edge cases (div by zero, sqrt of negative)
# - Consider using numpy.errstate or warning filters
# - Test the numerical stability with slopes close to zero or infinity

from dataclasses import dataclass
from scipy import stats
from typing import Protocol, cast

import numpy as np


@dataclass(frozen=True)
class Point:
    """A point in the plane."""

    x: float
    y: float


@dataclass(frozen=True)
class Box:
    """The box inside which we're drawing."""

    min: Point = Point(0, 0)
    max: Point = Point(10000, 10000)


@dataclass(frozen=True)
class Segment:
    """A segment connecting points in the bounding box boundary."""

    start: Point
    end: Point


class Regression(Protocol):
    """Statistics from linear regression computation.

    Attributes:
        slope: m in y = mx + b
        intercept: b in y = mx + b
        rvalue: correlation coefficient (-1 to 1, higher absolute value = better fit)
    """

    slope: float
    intercept: float
    rvalue: float


@dataclass(frozen=True)
class Result:
    """Bundles the result segment along with the regression results."""

    segment: Segment
    regr_horiz: Regression | None
    regr_vert: Regression | None


@dataclass(frozen=True)
class _ResultComputer:
    """Internal class that implements the bounding box intersection algorithm.

    Theory:
    -------
    1. Linear regression finds a line y = mx + b (or x = my + b) that best fits
       the input points by minimizing the squared distances between points and line.

    2. We compute two regressions because:
       - y = mx + b regression minimizes vertical distances
       - x = my + b regression minimizes horizontal distances
       When points form a near-vertical line, the x = my + b regression might
       give better results, which we detect using the r-value (correlation coefficient).

    3. For any line equation (y = mx + b), we find intersections with the box by:
       - Plugging in x-coordinates of vertical edges to find y intersections
       - Solving for x when y equals horizontal edges' y-coordinates

    Special cases:
    -------------
    1. Horizontal line (slope = 0):
       - Line equation: y = b (constant)
       - Intersects left and right edges of box at y = b

    2. Vertical line (1/slope = 0):
       - Line equation: x = b (constant)
       - Intersects top and bottom edges of box at x = b

    3. No regression possible:
       - Empty input
    """

    box: Box
    regr_horiz: Regression | None
    regr_vert: Regression | None

    def compute_result(self) -> Result | None:
        """Compute intersection points with bounding box.

        Algorithm:
        1. If no regression is possible -> return None
        2. If only one regression exists -> use it
        3. If both exist -> use the one with higher r-value (better fit)
        """

        # No regressions
        if not self.regr_horiz and not self.regr_vert:
            return None

        # Just one regression
        if self.regr_horiz and not self.regr_vert:
            return self._compute_horizontal_result(self.regr_horiz)
        if not self.regr_horiz and self.regr_vert:
            return self._compute_vertical_result(self.regr_vert)

        # We have both
        assert self.regr_horiz and self.regr_vert
        if self.regr_horiz.rvalue >= self.regr_vert.rvalue:
            return self._compute_horizontal_result(self.regr_horiz)
        return self._compute_vertical_result(self.regr_vert)

    def _compute_horizontal_result(self, regr: Regression) -> Result | None:
        """Handle y = mx + b regression result.

        Cases:
        1. Horizontal line (slope = 0): return line at y = intercept
        2. Normal case: find intersections with box
           - If no intersections or just 1: return None (line misses box)
           - If 2 intersections: return those points
           - If >2 intersections: take first and last (should not happen with convex box)
        """
        points = self._compute_intersections(regr.slope, regr.intercept)

        # No valid intersections with box
        if len(points) < 2:
            # Special case: horizontal line
            if abs(regr.slope) < 1e-10:  # Use small epsilon instead of exact 0
                return Result(
                    segment=Segment(
                        start=Point(self.box.min.x, regr.intercept),
                        end=Point(self.box.max.x, regr.intercept),
                    ),
                    regr_horiz=self.regr_horiz,
                    regr_vert=self.regr_vert,
                )
            return None  # Line misses box

        # Take first and last intersection points
        return Result(
            segment=Segment(start=points[0], end=points[-1]),
            regr_horiz=self.regr_horiz,
            regr_vert=self.regr_vert,
        )

    def _compute_vertical_result(self, regr: Regression) -> Result | None:
        """Handle x = my + b regression result.

        Cases:
        1. m ≈ 0: perfect vertical line x = b
        2. Otherwise: convert x = my + b to y = (1/m)x + (-b/m)

        Note: infinite slope should not occur in vertical regression
        """
        if abs(regr.slope) < 1e-10:  # Nearly vertical line
            return Result(
                segment=Segment(
                    start=Point(regr.intercept, self.box.min.y),
                    end=Point(regr.intercept, self.box.max.y),
                ),
                regr_horiz=self.regr_horiz,
                regr_vert=self.regr_vert,
            )

        # Normal case
        slope = 1.0 / regr.slope
        intercept = -regr.intercept / regr.slope

        points = self._compute_intersections(slope, intercept)
        if len(points) < 2:
            return None

        return Result(
            segment=Segment(start=points[0], end=points[-1]),
            regr_horiz=self.regr_horiz,
            regr_vert=self.regr_vert,
        )

    def _compute_intersections(self, slope: float, intercept: float) -> list[Point]:
        """Find where line y = mx + b intersects the bounding box.

        For line equation y = mx + b:
        1. At x = x0: Point(x0, mx0 + b)
        2. At y = y0: Point((y0 - b)/m, y0) if m ≠ 0

        Returns points where these intersections lie within box boundaries.
        """
        points = []

        # Left edge (x = min.x)
        y = slope * self.box.min.x + intercept
        if self.box.min.y <= y <= self.box.max.y:
            points.append(Point(self.box.min.x, y))

        # Right edge (x = max.x)
        y = slope * self.box.max.x + intercept
        if self.box.min.y <= y <= self.box.max.y:
            points.append(Point(self.box.max.x, y))

        # Bottom edge (y = min.y)
        if slope != 0:  # Avoid division by zero
            x = (self.box.min.y - intercept) / slope
            if self.box.min.x <= x <= self.box.max.x:
                points.append(Point(x, self.box.min.y))

        # Top edge (y = max.y)
        if slope != 0:  # Avoid division by zero
            x = (self.box.max.y - intercept) / slope
            if self.box.min.x <= x <= self.box.max.x:
                points.append(Point(x, self.box.max.y))

        return points


def compute(x_values: np.ndarray, y_values: np.ndarray, box: Box) -> Result | None:
    """Compute regression line and intersections.

    Regression can fail when:
    - No points provided

    We compute both y = mx + b and x = my + b regressions because:
    - y = mx + b minimizes vertical distances (better for shallow slopes)
    - x = my + b minimizes horizontal distances (better for steep slopes)

    We gracefully handle vertical and hortizontal lines.
    """

    # 0. First check if all points are outside box
    inside_box = False
    for x, y in zip(x_values, y_values):
        if (box.min.x <= x <= box.max.x and
            box.min.y <= y <= box.max.y):
            inside_box = True
            break

    if not inside_box:
        return None

    # 1. compute a regression line assuming x is independent
    # and fail for lack of points or collinearity
    regr_horiz: Regression | None = None
    try:
        regr_horiz = cast(Regression, stats.linregress(x_values, y_values))
    except ValueError:
        pass

    # 2. same as above but assuming y is independent
    regr_vert: Regression | None = None
    try:
        regr_vert = cast(Regression, stats.linregress(y_values, x_values))
    except ValueError:
        pass

    # 3. defer to the _ResultComputer to compute the result
    rc = _ResultComputer(box, regr_horiz, regr_vert)
    return rc.compute_result()
