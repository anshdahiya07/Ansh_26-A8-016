import cv2
import numpy as np
import os


# ============================================================
# LANE DETECTION FOR MULTIPLE IMAGES
# Simple + stable version
# ============================================================


def process_image(input_path, output_path):

    # --------------------------------------------------------
    # 1. READ IMAGE
    # --------------------------------------------------------

    image = cv2.imread(input_path)

    if image is None:
        print("Could not read:", input_path)
        return

    height, width = image.shape[:2]


    # --------------------------------------------------------
    # 2. GRAYSCALE + BLUR
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )


    # --------------------------------------------------------
    # 3. CANNY EDGES
    # --------------------------------------------------------

    edges = cv2.Canny(
        blur,
        50,
        150
    )


    # --------------------------------------------------------
    # 4. ROI
    # --------------------------------------------------------

    mask = np.zeros_like(edges)

    roi_points = np.array([
        (0, height),
        (int(width * 0.30), int(height * 0.43)),
        (int(width * 0.70), int(height * 0.43)),
        (width, height)
    ], dtype=np.int32)

    cv2.fillPoly(
        mask,
        [roi_points],
        255
    )

    roi = cv2.bitwise_and(
        edges,
        mask
    )


    # --------------------------------------------------------
    # 5. HSV FOR LANE COLORS
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )


    # Yellow
    yellow_lower = np.array(
        [10, 70, 70],
        dtype=np.uint8
    )

    yellow_upper = np.array(
        [45, 255, 255],
        dtype=np.uint8
    )

    yellow_mask = cv2.inRange(
        hsv,
        yellow_lower,
        yellow_upper
    )


    # White
    white_lower = np.array(
        [0, 0, 150],
        dtype=np.uint8
    )

    white_upper = np.array(
        [180, 90, 255],
        dtype=np.uint8
    )

    white_mask = cv2.inRange(
        hsv,
        white_lower,
        white_upper
    )


    # Apply ROI to color masks

    yellow_mask = cv2.bitwise_and(
        yellow_mask,
        mask
    )

    white_mask = cv2.bitwise_and(
        white_mask,
        mask
    )


    # --------------------------------------------------------
    # 6. COMBINE EDGES WITH LANE COLORS
    #
    # Give preference to actual yellow/white markings.
    # --------------------------------------------------------

    color_mask = cv2.bitwise_or(
        yellow_mask,
        white_mask
    )


    color_edges = cv2.Canny(
        color_mask,
        30,
        100
    )


    # Combine normal edges and color edges
    combined = cv2.bitwise_or(
        roi,
        color_edges
    )


    # --------------------------------------------------------
    # 7. HOUGH LINES
    # --------------------------------------------------------

    lines = cv2.HoughLinesP(
        combined,
        1,
        np.pi / 180,
        threshold=40,
        minLineLength=max(
            40,
            int(width * 0.06)
        ),
        maxLineGap=max(
            50,
            int(width * 0.08)
        )
    )


    # --------------------------------------------------------
    # 8. COLLECT LEFT AND RIGHT LINES
    # --------------------------------------------------------

    left_lines = []
    right_lines = []


    if lines is not None:

        for line in lines:

            x1, y1, x2, y2 = map(
                int,
                line[0]
            )


            dx = x2 - x1
            dy = y2 - y1


            if dx == 0:
                continue


            slope = dy / dx


            # Ignore nearly horizontal lines
            if abs(slope) < 0.35:
                continue


            # Ignore almost vertical lines
            if abs(slope) > 4.0:
                continue


            # Midpoint
            mid_x = (
                x1 + x2
            ) / 2

            mid_y = (
                y1 + y2
            ) / 2


            if mid_y < height * 0.43:
                continue


            # ------------------------------------------------
            # Calculate x position at bottom
            # ------------------------------------------------

            if dy == 0:
                continue


            bottom_y = height - 1


            x_bottom = (
                x1
                +
                (bottom_y - y1)
                *
                dx
                /
                dy
            )


            # Reject impossible lines
            if (
                x_bottom < -width * 0.25
                or
                x_bottom > width * 1.25
            ):
                continue


            length = np.sqrt(
                dx * dx + dy * dy
            )


            # ------------------------------------------------
            # LEFT
            # ------------------------------------------------

            if slope < -0.35:

                # Must finish on left/centre
                if x_bottom < width * 0.58:

                    left_lines.append(
                        (
                            x_bottom,
                            length,
                            slope,
                            x1,
                            y1,
                            x2,
                            y2
                        )
                    )


            # ------------------------------------------------
            # RIGHT
            # ------------------------------------------------

            elif slope > 0.35:

                # Must finish on right/centre
                if x_bottom > width * 0.42:

                    right_lines.append(
                        (
                            x_bottom,
                            length,
                            slope,
                            x1,
                            y1,
                            x2,
                            y2
                        )
                    )


    # ========================================================
    # 9. SELECT BEST LEFT LINE
    # ========================================================

    def select_left(lines_list):

        if len(lines_list) == 0:
            return None


        # Prefer lines around the expected
        # left lane position.

        expected_x = width * 0.25


        def score(item):

            x_bottom = item[0]
            length = item[1]

            distance = abs(
                x_bottom - expected_x
            )

            position_score = max(
                0,
                1 - distance / width
            )

            length_score = min(
                length / (width * 0.25),
                1
            )

            return (
                position_score * 0.65
                +
                length_score * 0.35
            )


        return max(
            lines_list,
            key=score
        )


    # ========================================================
    # 10. SELECT BEST RIGHT LINE
    # ========================================================

    def select_right(lines_list):

        if len(lines_list) == 0:
            return None


        expected_x = width * 0.78


        def score(item):

            x_bottom = item[0]
            length = item[1]

            distance = abs(
                x_bottom - expected_x
            )

            position_score = max(
                0,
                1 - distance / width
            )

            length_score = min(
                length / (width * 0.25),
                1
            )

            return (
                position_score * 0.65
                +
                length_score * 0.35
            )


        return max(
            lines_list,
            key=score
        )


    best_left = select_left(
        left_lines
    )

    best_right = select_right(
        right_lines
    )


    # ========================================================
    # 11. CREATE OUTPUT
    # ========================================================

    output = image.copy()


    # If we don't have both lines,
    # don't create a wrong green area.

    if (
        best_left is None
        or
        best_right is None
    ):

        print(
            "Skipped - not enough lane lines:",
            os.path.basename(input_path)
        )

        cv2.imwrite(
            output_path,
            output
        )

        return


    # ========================================================
    # 12. EXTRACT BEST LINES
    # ========================================================

    left_x_bottom = best_left[0]
    right_x_bottom = best_right[0]


    # --------------------------------------------------------
    # IMPORTANT SANITY CHECK
    # --------------------------------------------------------

    lane_width = (
        right_x_bottom
        -
        left_x_bottom
    )


    # Reject clearly wrong detections

    if (
        lane_width < width * 0.20
        or
        lane_width > width * 0.90
    ):

        print(
            "Skipped - incorrect lane width:",
            os.path.basename(input_path)
        )

        cv2.imwrite(
            output_path,
            output
        )

        return


    # ========================================================
    # 13. CREATE STRAIGHT LANE BOUNDARIES
    # ========================================================

    def make_line_points(line):

        x_bottom = line[0]
        slope = line[2]

        y_bottom = height - 1

        # Don't go too close to horizon
        y_top = int(
            height * 0.43
        )


        # x1 = x2 - (y2-y1)/slope
        x_top = (
            x_bottom
            -
            (y_bottom - y_top)
            /
            slope
        )


        return (
            int(x_top),
            y_top,
            int(x_bottom),
            y_bottom
        )


    left_points = make_line_points(
        best_left
    )

    right_points = make_line_points(
        best_right
    )


    # ========================================================
    # 14. FINAL GEOMETRY CHECK
    # ========================================================

    lx_top = left_points[0]
    lx_bottom = left_points[2]

    rx_top = right_points[0]
    rx_bottom = right_points[2]


    # Left must remain left of right

    if (
        lx_top >= rx_top
        or
        lx_bottom >= rx_bottom
    ):

        print(
            "Skipped - invalid geometry:",
            os.path.basename(input_path)
        )

        cv2.imwrite(
            output_path,
            output
        )

        return


    # --------------------------------------------------------
    # Top lane should not be too wide
    # --------------------------------------------------------

    top_width = (
        rx_top
        -
        lx_top
    )


    if top_width > width * 0.55:

        print(
            "Skipped - top lane too wide:",
            os.path.basename(input_path)
        )

        cv2.imwrite(
            output_path,
            output
        )

        return


    # ========================================================
    # 15. CREATE GREEN POLYGON
    # ========================================================

    polygon = np.array([
        (lx_top, left_points[1]),
        (rx_top, right_points[1]),
        (rx_bottom, right_points[3]),
        (lx_bottom, left_points[3])
    ], dtype=np.int32)


    # ========================================================
    # 16. GREEN TRANSPARENT AREA
    # ========================================================

    overlay = output.copy()


    cv2.fillPoly(
        overlay,
        [polygon],
        (0, 255, 0)
    )


    output = cv2.addWeighted(
        overlay,
        0.25,
        output,
        0.75,
        0
    )


    # ========================================================
    # 17. DRAW LEFT GREEN LINE
    # ========================================================

    cv2.line(
        output,
        (
            left_points[0],
            left_points[1]
        ),
        (
            left_points[2],
            left_points[3]
        ),
        (0, 255, 0),
        5
    )


    # ========================================================
    # 18. DRAW RIGHT GREEN LINE
    # ========================================================

    cv2.line(
        output,
        (
            right_points[0],
            right_points[1]
        ),
        (
            right_points[2],
            right_points[3]
        ),
        (0, 255, 0),
        5
    )


    # ========================================================
    # 19. SAVE
    # ========================================================

    success = cv2.imwrite(
        output_path,
        output
    )


    if success:

        print(
            "Processed:",
            os.path.basename(input_path)
        )

    else:

        print(
            "ERROR saving:",
            os.path.basename(input_path)
        )


# ============================================================
# MAIN
# ============================================================


input_folder = "input"
output_folder = "output"


os.makedirs(
    output_folder,
    exist_ok=True
)


# Supported image formats

extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
)


# Find all images

image_files = [
    f
    for f in os.listdir(input_folder)
    if f.lower().endswith(
        extensions
    )
]


# Sort 1,2,3...10 correctly

def sort_key(filename):

    name = os.path.splitext(
        filename
    )[0]

    try:
        return int(name)

    except ValueError:
        return name.lower()


image_files.sort(
    key=sort_key
)


# ============================================================
# PROCESS ALL IMAGES
# ============================================================

print()
print(
    "======================================"
)

print(
    "LANE DETECTION STARTED"
)

print(
    "Images found:",
    len(image_files)
)

print(
    "======================================"
)

print()


for filename in image_files:

    input_path = os.path.join(
        input_folder,
        filename
    )


    name = os.path.splitext(
        filename
    )[0]


    output_filename = (
        name
        +
        "_detected.png"
    )


    output_path = os.path.join(
        output_folder,
        output_filename
    )


    process_image(
        input_path,
        output_path
    )


# ============================================================
# FINISHED
# ============================================================

print()
print(
    "======================================"
)

print(
    "ALL IMAGES PROCESSED"
)

print(
    "Check the output folder."
)

print(
    "======================================"
)