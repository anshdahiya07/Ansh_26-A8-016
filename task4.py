import cv2
import numpy as np
import heapq
import math
import os
import glob


# ============================================================
# SETTINGS
# ============================================================

INPUT_FOLDER = "image4"
OUTPUT_FOLDER = "output4"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Work on a smaller grid so A* is fast
GRID_SCALE = 4

# Safety margins
ROAD_MARGIN = 8
OBSTACLE_MARGIN = 12


# ============================================================
# CHECKPOINTS FOR THE 10 PROVIDED IMAGES
# Coordinates are (x, y)
# Last point repeats the first point -> one complete loop
# ============================================================

CHECKPOINTS = {

    1: [
        (1000, 700),
        (960, 790),
        (850, 860),
        (720, 920),
        (580, 930),
        (430, 880),
        (320, 800),
        (260, 680),
        (260, 550),
        (290, 450),
        (370, 400),
        (470, 420),
        (560, 480),
        (650, 460),
        (730, 390),
        (790, 320),
        (850, 360),
        (920, 470),
        (980, 590),
        (1000, 700)
    ],

    2: [
        (1070, 690),
        (1030, 800),
        (930, 890),
        (800, 960),
        (650, 1000),
        (500, 980),
        (370, 920),
        (260, 820),
        (210, 700),
        (230, 580),
        (250, 470),
        (220, 360),
        (320, 320),
        (450, 310),
        (570, 250),
        (690, 300),
        (820, 370),
        (930, 450),
        (1010, 550),
        (1070, 690)
    ],

    3: [
        (870, 700),
        (820, 770),
        (730, 800),
        (650, 860),
        (570, 920),
        (470, 930),
        (370, 890),
        (290, 820),
        (230, 730),
        (220, 620),
        (240, 520),
        (290, 440),
        (370, 410),
        (460, 430),
        (550, 450),
        (650, 420),
        (750, 370),
        (840, 390),
        (890, 500),
        (900, 610),
        (870, 700)
    ],

    4: [
        (880, 700),
        (790, 720),
        (690, 750),
        (650, 830),
        (650, 930),
        (600, 1020),
        (510, 1070),
        (420, 1030),
        (350, 950),
        (320, 850),
        (330, 750),
        (360, 650),
        (380, 550),
        (340, 460),
        (300, 360),
        (410, 390),
        (530, 430),
        (650, 390),
        (760, 330),
        (850, 290),
        (920, 350),
        (950, 460),
        (940, 570),
        (880, 700)
    ],

    5: [
        (880, 700),
        (790, 720),
        (690, 750),
        (650, 830),
        (650, 930),
        (600, 1020),
        (510, 1070),
        (420, 1030),
        (350, 950),
        (320, 850),
        (330, 750),
        (360, 650),
        (380, 550),
        (340, 460),
        (300, 360),
        (410, 390),
        (530, 430),
        (650, 390),
        (760, 330),
        (850, 290),
        (920, 350),
        (950, 460),
        (940, 570),
        (880, 700)
    ],

    6: [
        (900, 550),
        (980, 640),
        (1030, 730),
        (1000, 820),
        (900, 890),
        (780, 950),
        (650, 990),
        (520, 990),
        (390, 950),
        (270, 880),
        (190, 800),
        (160, 700),
        (180, 600),
        (230, 500),
        (300, 430),
        (390, 390),
        (500, 380),
        (620, 380),
        (740, 410),
        (840, 470),
        (900, 550)
    ],

    7: [
        (250, 600),
        (240, 520),
        (270, 440),
        (330, 360),
        (420, 300),
        (520, 250),
        (630, 210),
        (750, 200),
        (850, 230),
        (920, 300),
        (950, 400),
        (940, 500),
        (950, 610),
        (980, 710),
        (950, 800),
        (860, 850),
        (750, 880),
        (630, 880),
        (500, 860),
        (380, 830),
        (290, 760),
        (240, 680),
        (250, 600)
    ],

    8: [
        (650, 990),
        (550, 980),
        (450, 950),
        (360, 900),
        (300, 830),
        (280, 750),
        (300, 670),
        (350, 580),
        (400, 500),
        (450, 420),
        (500, 350),
        (540, 270),
        (600, 200),
        (670, 190),
        (720, 250),
        (760, 340),
        (820, 410),
        (900, 430),
        (960, 400),
        (930, 490),
        (860, 560),
        (820, 650),
        (850, 730),
        (930, 790),
        (970, 860),
        (930, 930),
        (820, 970),
        (720, 990),
        (650, 990)
    ],

    9: [
        (880, 700),
        (900, 780),
        (880, 860),
        (830, 920),
        (750, 960),
        (650, 990),
        (540, 990),
        (430, 970),
        (340, 920),
        (290, 850),
        (260, 760),
        (250, 660),
        (270, 560),
        (300, 470),
        (350, 390),
        (440, 340),
        (540, 300),
        (650, 260),
        (760, 230),
        (850, 220),
        (920, 270),
        (950, 360),
        (950, 460),
        (930, 560),
        (900, 650),
        (880, 700)
    ],

    10: [
        (330, 830),
        (270, 800),
        (230, 740),
        (210, 650),
        (230, 570),
        (280, 500),
        (350, 460),
        (430, 430),
        (500, 380),
        (530, 300),
        (600, 250),
        (710, 220),
        (820, 220),
        (910, 250),
        (960, 330),
        (950, 430),
        (920, 540),
        (950, 640),
        (1020, 720),
        (1080, 800),
        (1060, 870),
        (980, 900),
        (860, 920),
        (740, 930),
        (650, 910),
        (560, 870),
        (470, 850),
        (390, 830),
        (330, 830)
    ]
}


# ============================================================
# ROAD DETECTION
# ============================================================

def detect_road(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(
        gray,
        (11, 11),
        0
    )

    # Road in these supplied images is slightly brighter
    # than the background.
    _, mask = cv2.threshold(
        blurred,
        91,
        255,
        cv2.THRESH_BINARY
    )

    # Join small breaks
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (17, 17)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # --------------------------------------------------------
    # Find the largest contour instead of
    # connectedComponentsWithStats().
    #
    # This avoids the Pylance/OpenCV typing problem you had.
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    road = np.zeros(
        gray.shape,
        dtype=np.uint8
    )

    if not contours:
        return road

    largest = max(
        contours,
        key=cv2.contourArea
    )

    cv2.drawContours(
        road,
        [largest],
        -1,
        255,
        thickness=cv2.FILLED
    )

    # Keep route slightly away from road edge
    erosion_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            ROAD_MARGIN * 2 + 1,
            ROAD_MARGIN * 2 + 1
        )
    )

    road = cv2.erode(
        road,
        erosion_kernel,
        iterations=1
    )

    return road


# ============================================================
# REMOVE SMALL BLOBS
# ============================================================

def remove_small_components(mask, minimum_area=60):

    result = np.zeros_like(
        mask,
        dtype=np.uint8
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area >= minimum_area:

            cv2.drawContours(
                result,
                [contour],
                -1,
                255,
                cv2.FILLED
            )

    return result


# ============================================================
# OBSTACLE / POTHOLE DETECTION
# ============================================================

def detect_obstacles(image, road):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]


    # --------------------------------------------------------
    # COLORED OBSTACLES
    # --------------------------------------------------------

    colored = cv2.inRange(
        saturation,
        35,
        255
    )


    # --------------------------------------------------------
    # DARK POTHOLES
    # --------------------------------------------------------

    dark = cv2.inRange(
        gray,
        0,
        75
    )


    # Prevent general dark background from becoming obstacle
    dark = cv2.bitwise_and(
        dark,
        road
    )


    colored = cv2.bitwise_and(
        colored,
        road
    )


    # Combine obstacle types
    obstacle_mask = cv2.bitwise_or(
        colored,
        dark
    )


    # Clean noise
    obstacle_mask = cv2.morphologyEx(
        obstacle_mask,
        cv2.MORPH_OPEN,
        np.ones(
            (3, 3),
            dtype=np.uint8
        )
    )


    obstacle_mask = cv2.morphologyEx(
        obstacle_mask,
        cv2.MORPH_CLOSE,
        np.ones(
            (7, 7),
            dtype=np.uint8
        )
    )


    obstacle_mask = remove_small_components(
        obstacle_mask,
        minimum_area=60
    )


    # --------------------------------------------------------
    # SAFETY BUFFER
    # --------------------------------------------------------

    safety_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            OBSTACLE_MARGIN * 2 + 1,
            OBSTACLE_MARGIN * 2 + 1
        )
    )


    obstacle_mask = cv2.dilate(
        obstacle_mask,
        safety_kernel,
        iterations=1
    )


    # Again restrict to road
    obstacle_mask = cv2.bitwise_and(
        obstacle_mask,
        road
    )

    return obstacle_mask


# ============================================================
# CREATE FREE SPACE
# ============================================================

def create_free_space(
    road,
    obstacles
):

    free = road.copy()

    free[
        obstacles > 0
    ] = 0

    return free


# ============================================================
# FIND NEAREST SAFE PIXEL
# ============================================================

def nearest_safe_point(
    point,
    free_space
):

    x = int(point[0])
    y = int(point[1])

    height, width = (
        free_space.shape
    )


    if (
        0 <= x < width
        and
        0 <= y < height
        and
        free_space[y, x] > 0
    ):
        return (x, y)


    # Search outwards instead of calculating
    # distance to every image pixel.

    max_radius = max(
        width,
        height
    )


    for radius in range(
        1,
        max_radius
    ):

        x1 = max(
            0,
            x - radius
        )

        x2 = min(
            width - 1,
            x + radius
        )

        y1 = max(
            0,
            y - radius
        )

        y2 = min(
            height - 1,
            y + radius
        )


        # top/bottom
        for xx in range(
            x1,
            x2 + 1
        ):

            if free_space[y1, xx] > 0:
                return (xx, y1)

            if free_space[y2, xx] > 0:
                return (xx, y2)


        # left/right
        for yy in range(
            y1,
            y2 + 1
        ):

            if free_space[yy, x1] > 0:
                return (x1, yy)

            if free_space[yy, x2] > 0:
                return (x2, yy)


    return None


# ============================================================
# HEURISTIC
# ============================================================

def heuristic(
    a,
    b
):

    return math.hypot(
        float(a[0] - b[0]),
        float(a[1] - b[1])
    )


# ============================================================
# A* ALGORITHM
# ============================================================

def a_star(
    free_space,
    start,
    goal
):

    height, width = (
        free_space.shape
    )


    # --------------------------------------------------------
    # Downscale navigation grid
    # --------------------------------------------------------

    grid_width = max(
        1,
        width // GRID_SCALE
    )

    grid_height = max(
        1,
        height // GRID_SCALE
    )


    grid = cv2.resize(
        free_space,
        (
            grid_width,
            grid_height
        ),
        interpolation=cv2.INTER_NEAREST
    )


    grid = (
        grid > 0
    )


    start_grid = (
        int(start[0] // GRID_SCALE),
        int(start[1] // GRID_SCALE)
    )

    goal_grid = (
        int(goal[0] // GRID_SCALE),
        int(goal[1] // GRID_SCALE)
    )


    # Keep coordinates in range
    start_grid = (
        min(
            max(start_grid[0], 0),
            grid_width - 1
        ),
        min(
            max(start_grid[1], 0),
            grid_height - 1
        )
    )


    goal_grid = (
        min(
            max(goal_grid[0], 0),
            grid_width - 1
        ),
        min(
            max(goal_grid[1], 0),
            grid_height - 1
        )
    )


    # Make start/goal valid if resizing moved them
    if not grid[
        start_grid[1],
        start_grid[0]
    ]:

        temp = (
            grid.astype(np.uint8)
            * 255
        )

        safe = nearest_safe_point(
            start_grid,
            temp
        )

        if safe is None:
            return []

        start_grid = safe


    if not grid[
        goal_grid[1],
        goal_grid[0]
    ]:

        temp = (
            grid.astype(np.uint8)
            * 255
        )

        safe = nearest_safe_point(
            goal_grid,
            temp
        )

        if safe is None:
            return []

        goal_grid = safe


    # --------------------------------------------------------
    # A* data
    # --------------------------------------------------------

    open_heap = []

    counter = 0


    heapq.heappush(
        open_heap,
        (
            0.0,
            counter,
            start_grid
        )
    )


    came_from = {
        start_grid: None
    }


    g_score = {
        start_grid: 0.0
    }


    closed = set()


    movements = [

        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),

        (-1, -1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0))
    ]


    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    while open_heap:

        _, _, current = heapq.heappop(
            open_heap
        )


        if current in closed:
            continue


        if current == goal_grid:
            break


        closed.add(
            current
        )


        cx, cy = current


        for dx, dy, step_cost in movements:

            nx = cx + dx
            ny = cy + dy


            if not (
                0 <= nx < grid_width
                and
                0 <= ny < grid_height
            ):
                continue


            if not grid[
                ny,
                nx
            ]:
                continue


            # Avoid cutting diagonally through obstacle corners
            if (
                dx != 0
                and
                dy != 0
            ):

                if (
                    not grid[cy, nx]
                    or
                    not grid[ny, cx]
                ):
                    continue


            neighbour = (
                nx,
                ny
            )


            tentative_g = (
                g_score[current]
                +
                float(step_cost)
            )


            old_g = g_score.get(
                neighbour,
                float("inf")
            )


            if tentative_g < old_g:

                came_from[
                    neighbour
                ] = current


                g_score[
                    neighbour
                ] = tentative_g


                f_score = (
                    tentative_g
                    +
                    heuristic(
                        neighbour,
                        goal_grid
                    )
                )


                counter += 1


                heapq.heappush(
                    open_heap,
                    (
                        float(f_score),
                        counter,
                        neighbour
                    )
                )


    # --------------------------------------------------------
    # No path
    # --------------------------------------------------------

    if goal_grid not in came_from:

        return []


    # --------------------------------------------------------
    # Reconstruct
    # --------------------------------------------------------

    grid_path = []

    current = goal_grid


    while current is not None:

        grid_path.append(
            current
        )

        current = came_from[
            current
        ]


    grid_path.reverse()


    # Back to original image coordinates
    path = []


    for gx, gy in grid_path:

        px = int(
            gx * GRID_SCALE
            +
            GRID_SCALE // 2
        )

        py = int(
            gy * GRID_SCALE
            +
            GRID_SCALE // 2
        )


        px = min(
            px,
            width - 1
        )

        py = min(
            py,
            height - 1
        )


        path.append(
            (px, py)
        )


    return path


# ============================================================
# PATH SIMPLIFICATION
# ============================================================

def line_is_safe(
    p1,
    p2,
    free_space
):

    line_mask = np.zeros_like(
        free_space,
        dtype=np.uint8
    )


    cv2.line(
        line_mask,
        p1,
        p2,
        255,
        2
    )


    bad = (
        (line_mask > 0)
        &
        (free_space == 0)
    )


    return not np.any(
        bad
    )


def simplify_path(
    path,
    free_space
):

    if len(path) <= 2:
        return path


    simplified = [
        path[0]
    ]


    index = 0


    while index < len(path) - 1:

        farthest = index + 1


        for j in range(
            len(path) - 1,
            index,
            -1
        ):

            if line_is_safe(
                path[index],
                path[j],
                free_space
            ):

                farthest = j
                break


        simplified.append(
            path[farthest]
        )


        index = farthest


    return simplified


# ============================================================
# GET IMAGE NUMBER
# ============================================================

def get_image_number(
    filepath
):

    filename = os.path.basename(
        filepath
    )


    stem = os.path.splitext(
        filename
    )[0]


    try:
        return int(stem)

    except ValueError:
        return None


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    filepath
):

    image_number = get_image_number(
        filepath
    )


    if image_number is None:

        print(
            "Skipping file:",
            filepath
        )

        return


    if image_number not in CHECKPOINTS:

        print(
            f"No checkpoints for image "
            f"{image_number}"
        )

        return


    image = cv2.imread(
        filepath
    )


    if image is None:

        print(
            "Could not read:",
            filepath
        )

        return


    print()
    print(
        "================================"
    )

    print(
        f"IMAGE {image_number}"
    )

    print(
        "================================"
    )


    # --------------------------------------------------------
    # Detect road
    # --------------------------------------------------------

    road = detect_road(
        image
    )


    # --------------------------------------------------------
    # Detect obstacles
    # --------------------------------------------------------

    obstacles = detect_obstacles(
        image,
        road
    )


    # --------------------------------------------------------
    # Safe map
    # --------------------------------------------------------

    free_space = create_free_space(
        road,
        obstacles
    )


    # --------------------------------------------------------
    # Correct checkpoints to nearest safe position
    # --------------------------------------------------------

    checkpoints = []


    for point in CHECKPOINTS[
        image_number
    ]:

        safe_point = nearest_safe_point(
            point,
            free_space
        )


        if safe_point is None:

            print(
                "Could not find safe point "
                "near checkpoint:",
                point
            )

            continue


        checkpoints.append(
            safe_point
        )


    if len(checkpoints) < 2:

        print(
            "Not enough valid checkpoints."
        )

        return


    # Force loop closure
    checkpoints[-1] = checkpoints[0]


    # --------------------------------------------------------
    # A* between checkpoints
    # --------------------------------------------------------

    complete_path = []

    failures = 0


    for i in range(
        len(checkpoints) - 1
    ):

        start = checkpoints[i]

        goal = checkpoints[i + 1]


        segment = a_star(
            free_space,
            start,
            goal
        )


        if not segment:

            print(
                "No path:",
                i + 1,
                "->",
                i + 2
            )

            failures += 1
            continue


        segment = simplify_path(
            segment,
            free_space
        )


        if complete_path:

            complete_path.extend(
                segment[1:]
            )

        else:

            complete_path.extend(
                segment
            )


    # ========================================================
    # DRAW RESULT
    # ========================================================

    output = image.copy()


    # --------------------------------------------------------
    # Road safe boundary
    # --------------------------------------------------------

    road_contours, _ = cv2.findContours(
        road,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    cv2.drawContours(
        output,
        road_contours,
        -1,
        (255, 255, 0),
        2
    )


    # --------------------------------------------------------
    # Obstacle safety zones
    # --------------------------------------------------------

    obstacle_contours, _ = cv2.findContours(
        obstacles,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    cv2.drawContours(
        output,
        obstacle_contours,
        -1,
        (0, 0, 255),
        2
    )


    # --------------------------------------------------------
    # Final safe path
    # --------------------------------------------------------

    if len(complete_path) >= 2:

        path_array = np.asarray(
            complete_path,
            dtype=np.int32
        ).reshape(
            (-1, 1, 2)
        )


        cv2.polylines(
            output,
            [path_array],
            False,
            (0, 255, 0),
            6,
            cv2.LINE_AA
        )


    # --------------------------------------------------------
    # Checkpoints
    # --------------------------------------------------------

    for index, point in enumerate(
        checkpoints[:-1],
        start=1
    ):

        cv2.circle(
            output,
            point,
            4,
            (255, 0, 255),
            -1
        )


    # --------------------------------------------------------
    # Start
    # --------------------------------------------------------

    start = checkpoints[0]


    cv2.circle(
        output,
        start,
        10,
        (0, 255, 0),
        -1
    )


    cv2.putText(
        output,
        "START / GOAL",
        (
            start[0] + 12,
            start[1] - 10
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        output,
        "SAFE A* LOOP",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )


    # --------------------------------------------------------
    # Save output
    # --------------------------------------------------------

    output_path = os.path.join(
        OUTPUT_FOLDER,
        f"path_{image_number}.jpeg"
    )


    success = cv2.imwrite(
        output_path,
        output
    )


    if success:

        print(
            "Saved:",
            output_path
        )

    else:

        print(
            "Could not save output."
        )


    print(
        "Path points:",
        len(complete_path)
    )

    print(
        "Failed segments:",
        failures
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "================================"
    )

    print(
        "AERIAL SAFE PATH PLANNING"
    )

    print(
        "Checkpoint + A* Method"
    )

    print(
        "================================"
    )


    files = []

    extensions = [
        "*.jpeg",
        "*.jpg",
        "*.png"
    ]


    for extension in extensions:

        files.extend(
            glob.glob(
                os.path.join(
                    INPUT_FOLDER,
                    extension
                )
            )
        )


    files = sorted(
        files,
        key=lambda path: (
            get_image_number(path)
            if get_image_number(path)
            is not None
            else 999999
        )
    )


    if not files:

        print()
        print(
            "ERROR: No images found."
        )

        print(
            "Put images inside:",
            INPUT_FOLDER
        )

        return


    print()
    print(
        "Images found:",
        len(files)
    )


    for filepath in files:

        process_image(
            filepath
        )


    print()
    print(
        "================================"
    )

    print(
        "DONE"
    )

    print(
        "Check the output folder."
    )

    print(
        "GREEN = safe A* route"
    )

    print(
        "RED = obstacle safety area"
    )

    print(
        "CYAN = safe road boundary"
    )

    print(
        "================================"
    )


if __name__ == "__main__":
    main()