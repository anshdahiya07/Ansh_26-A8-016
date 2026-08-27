import cv2
import numpy as np


# ============================================
# 1. READ IMAGE
# ============================================

image = cv2.imread("1.png")

if image is None:
    print("Image not found!")
    exit()

height, width = image.shape[:2]


# ============================================
# 2. CONVERT IMAGE TO HSV
# ============================================

hsv = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2HSV
)


# ============================================
# 3. YELLOW MASK
#    Used for the left lane marking
# ============================================

lower_yellow = np.array(
    [15, 80, 80],
    dtype=np.uint8
)

upper_yellow = np.array(
    [40, 255, 255],
    dtype=np.uint8
)

yellow_mask = cv2.inRange(
    hsv,
    lower_yellow,
    upper_yellow
)


# ============================================
# 4. WHITE MASK
#    Used for the right lane marking
# ============================================

lower_white = np.array(
    [0, 0, 160],
    dtype=np.uint8
)

upper_white = np.array(
    [180, 90, 255],
    dtype=np.uint8
)

white_mask = cv2.inRange(
    hsv,
    lower_white,
    upper_white
)


# ============================================
# 5. CREATE ROI
# ============================================

roi_mask = np.zeros(
    (height, width),
    dtype=np.uint8
)

roi_points = np.array([
    (0, height),
    (300, int(height * 0.42)),
    (760, int(height * 0.42)),
    (width, height)
], dtype=np.int32)

cv2.fillPoly(
    roi_mask,
    [roi_points],
    255
)


# Apply ROI to yellow mask

yellow_mask = cv2.bitwise_and(
    yellow_mask,
    roi_mask
)


# Apply ROI to white mask

white_mask = cv2.bitwise_and(
    white_mask,
    roi_mask
)


# ============================================
# 6. REMOVE SMALL NOISE
# ============================================

kernel = np.ones(
    (5, 5),
    dtype=np.uint8
)

yellow_mask = cv2.morphologyEx(
    yellow_mask,
    cv2.MORPH_OPEN,
    kernel
)

white_mask = cv2.morphologyEx(
    white_mask,
    cv2.MORPH_OPEN,
    kernel
)


# ============================================
# 7. HOUGH TRANSFORM ON YELLOW MASK
# ============================================

yellow_lines = cv2.HoughLinesP(
    yellow_mask,
    1,
    np.pi / 180,
    threshold=30,
    minLineLength=40,
    maxLineGap=60
)


# ============================================
# 8. HOUGH TRANSFORM ON WHITE MASK
# ============================================

white_lines = cv2.HoughLinesP(
    white_mask,
    1,
    np.pi / 180,
    threshold=30,
    minLineLength=40,
    maxLineGap=60
)


# ============================================
# 9. FIND LEFT LANE LINE
# ============================================

left_candidates = []

if yellow_lines is not None:

    # Convert Hough output to simple
    # rows containing x1,y1,x2,y2
    yellow_data = yellow_lines.reshape(
        -1,
        4
    ).tolist()

    for values in yellow_data:

        x1, y1, x2, y2 = [
            int(value)
            for value in values
        ]

        if x2 == x1:
            continue

        # Calculate slope
        slope = (
            (y2 - y1)
            / (x2 - x1)
        )

        # Calculate midpoint
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2

        # Left lane should have
        # negative slope
        if slope < -0.5:

            # Keep left-side lines
            if mid_x < width * 0.55:

                # Ignore high lines
                if mid_y > height * 0.40:

                    left_candidates.append(
                        (slope, x1, y1, x2, y2)
                    )


# ============================================
# 10. FIND RIGHT LANE LINE
# ============================================

right_candidates = []

if white_lines is not None:

    # Convert Hough output to simple
    # rows containing x1,y1,x2,y2
    white_data = white_lines.reshape(
        -1,
        4
    ).tolist()

    for values in white_data:

        x1, y1, x2, y2 = [
            int(value)
            for value in values
        ]

        if x2 == x1:
            continue

        # Calculate slope
        slope = (
            (y2 - y1)
            / (x2 - x1)
        )

        # Calculate midpoint
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2

        # Right lane should have
        # positive slope
        if slope > 0.5:

            # Keep right-side lines
            if mid_x > width * 0.45:

                # Ignore high lines
                if mid_y > height * 0.40:

                    right_candidates.append(
                        (slope, x1, y1, x2, y2)
                    )


# ============================================
# 11. CALCULATE BEST LINE
# ============================================

def calculate_line(candidates):

    if len(candidates) == 0:
        return None

    slopes = []
    intercepts = []

    for slope, x1, y1, x2, y2 in candidates:

        # y = mx + b
        intercept = y1 - slope * x1

        slopes.append(slope)
        intercepts.append(intercept)

    # Median is more stable
    # than average
    slope = float(
        np.median(slopes)
    )

    intercept = float(
        np.median(intercepts)
    )

    return slope, intercept


left_line = calculate_line(
    left_candidates
)

right_line = calculate_line(
    right_candidates
)


# ============================================
# 12. CREATE LINE POINTS
# ============================================

def make_points(line):

    if line is None:
        return None

    slope, intercept = line

    # Top of lane
    y_top = int(
        height * 0.43
    )

    # Bottom of image
    y_bottom = height - 1

    # x = (y - b) / m

    x_top = int(
        (y_top - intercept)
        / slope
    )

    x_bottom = int(
        (y_bottom - intercept)
        / slope
    )

    return (
        x_top,
        y_top,
        x_bottom,
        y_bottom
    )


left_points = make_points(
    left_line
)

right_points = make_points(
    right_line
)


# ============================================
# 13. LIMIT RIGHT LINE
# ============================================

if right_points is not None:

    rx_top, ry_top, rx_bottom, ry_bottom = right_points

    # Keep the top point near the
    # actual white road marking
    rx_top = max(
        rx_top,
        int(width * 0.48)
    )

    rx_top = min(
        rx_top,
        int(width * 0.62)
    )

    # Keep bottom point on the
    # right road boundary
    rx_bottom = max(
        rx_bottom,
        int(width * 0.82)
    )

    rx_bottom = min(
        rx_bottom,
        int(width * 0.94)
    )

    right_points = (
        rx_top,
        ry_top,
        rx_bottom,
        ry_bottom
    )


# ============================================
# 14. CREATE OUTPUT
# ============================================

output = image.copy()


# ============================================
# 15. CREATE DRIVABLE AREA
# ============================================

if (
    left_points is not None
    and right_points is not None
):

    lx_top, ly_top, lx_bottom, ly_bottom = (
        left_points
    )

    rx_top, ry_top, rx_bottom, ry_bottom = (
        right_points
    )


    # ----------------------------------------
    # Create lane polygon
    # ----------------------------------------

    polygon = np.array([
        (lx_top, ly_top),
        (rx_top, ry_top),
        (rx_bottom, ry_bottom),
        (lx_bottom, ly_bottom)
    ], dtype=np.int32)


    # ----------------------------------------
    # Create overlay
    # ----------------------------------------

    overlay = output.copy()

    cv2.fillPoly(
        overlay,
        [polygon],
        (0, 255, 0)
    )


    # ----------------------------------------
    # Transparent green area
    # ----------------------------------------

    output = cv2.addWeighted(
        overlay,
        0.25,
        output,
        0.75,
        0
    )


    # ========================================
    # 16. DRAW LEFT GREEN LINE
    # ========================================

    cv2.line(
        output,
        (lx_top, ly_top),
        (lx_bottom, ly_bottom),
        (0, 255, 0),
        5
    )


    # ========================================
    # 17. DRAW RIGHT GREEN LINE
    # ========================================

    cv2.line(
        output,
        (rx_top, ry_top),
        (rx_bottom, ry_bottom),
        (0, 255, 0),
        5
    )


# ============================================
# 18. SAVE OUTPUT IMAGE
# ============================================

success = cv2.imwrite(
    "lane_output.png",
    output
)

if success:

    print(
        "Output saved successfully as "
        "lane_output.png"
    )

else:

    print(
        "Could not save output image!"
    )


# ============================================
# 19. DISPLAY OUTPUT
# ============================================

cv2.imshow(
    "Lane Detection",
    output
)


# ============================================
# 20. KEEP WINDOW OPEN
# ============================================

while True:

    key = cv2.waitKey(100) & 0xFF

    # Press any key to close
    if key != 255:
        break

    # Close using X
    if cv2.getWindowProperty(
        "Lane Detection",
        cv2.WND_PROP_VISIBLE
    ) < 1:
        break


# ============================================
# 21. CLOSE WINDOWS
# ============================================

cv2.destroyAllWindows()