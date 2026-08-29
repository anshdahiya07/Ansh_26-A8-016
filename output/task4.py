import cv2
import numpy as np
import heapq
import math
import os


# ============================================================
# FOLDER SETTINGS
# ============================================================

INPUT_FOLDER = "images4"
OUTPUT_FOLDER = "output4"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# CHECKPOINTS
# ============================================================
#
# The checkpoints are selected along the center of the track.
# They force A* to follow the complete track instead of
# taking a shortcut across the middle.
#
# Format:
# (x, y)
#
# The LAST checkpoint is the same as the START point so that
# one complete loop is generated.
# ============================================================

CHECKPOINTS = {

    # --------------------------------------------------------
    # IMAGE 1
    # --------------------------------------------------------
    1: [
        (1000, 690),
        (1000, 760),
        (900, 830),
        (800, 900),
        (700, 930),
        (580, 930),
        (450, 900),
        (330, 820),
        (270, 730),
        (260, 600),
        (270, 500),
        (320, 420),
        (420, 400),
        (520, 460),
        (620, 480),
        (720, 420),
        (800, 330),
        (870, 380),
        (930, 500),
        (980, 600),
        (1000, 690)
    ],


    # --------------------------------------------------------
    # IMAGE 2
    # --------------------------------------------------------
    2: [
        (1070, 690),
        (1050, 780),
        (980, 850),
        (850, 950),
        (700, 1000),
        (550, 1000),
        (400, 950),
        (280, 850),
        (210, 750),
        (220, 650),
        (250, 550),
        (210, 400),
        (300, 330),
        (450, 320),
        (580, 260),
        (700, 300),
        (820, 360),
        (930, 430),
        (1000, 520),
        (1050, 620),
        (1070, 690)
    ],


    # --------------------------------------------------------
    # IMAGE 3
    # --------------------------------------------------------
    3: [
        (870, 700),
        (820, 760),
        (720, 790),
        (650, 850),
        (600, 900),
        (500, 930),
        (400, 900),
        (320, 820),
        (250, 760),
        (220, 650),
        (230, 550),
        (260, 450),
        (350, 420),
        (450, 440),
        (550, 440),
        (650, 400),
        (760, 350),
        (850, 380),
        (900, 500),
        (900, 620),
        (870, 700)
    ],


    # --------------------------------------------------------
    # IMAGE 4
    # --------------------------------------------------------
    4: [
        (860, 700),
        (760, 710),
        (670, 730),
        (650, 800),
        (650, 900),
        (620, 1000),
        (520, 1080),
        (420, 1040),
        (350, 950),
        (320, 850),
        (330, 750),
        (360, 650),
        (380, 560),
        (330, 450),
        (300, 350),
        (430, 390),
        (560, 420),
        (700, 330),
        (820, 280),
        (900, 330),
        (950, 450),
        (950, 560),
        (900, 640),
        (860, 700)
    ],


    # --------------------------------------------------------
    # IMAGE 5
    # --------------------------------------------------------
    5: [
        (860, 700),
        (760, 710),
        (670, 730),
        (650, 800),
        (650, 900),
        (620, 1000),
        (520, 1080),
        (420, 1040),
        (350, 950),
        (320, 850),
        (330, 750),
        (360, 650),
        (380, 560),
        (330, 450),
        (300, 350),
        (430, 390),
        (560, 420),
        (700, 330),
        (820, 280),
        (900, 330),
        (950, 450),
        (950, 560),
        (900, 640),
        (860, 700)
    ],


    # --------------------------------------------------------
    # IMAGE 6
    # --------------------------------------------------------
    6: [
        (900, 550),
        (980, 650),
        (1040, 740),
        (1000, 820),
        (900, 900),
        (780, 960),
        (650, 1000),
        (500, 990),
        (350, 930),
        (220, 850),
        (150, 760),
        (170, 650),
        (230, 520),
        (300, 430),
        (380, 390),
        (500, 380),
        (650, 380),
        (780, 430),
        (880, 500),
        (900, 550)
    ],


    # --------------------------------------------------------
    # IMAGE 7
    # --------------------------------------------------------
    7: [
        (250, 600),
        (230, 520),
        (260, 440),
        (320, 360),
        (400, 300),
        (500, 250),
        (620, 210),
        (750, 200),
        (850, 220),
        (930, 280),
        (950, 380),
        (930, 500),
        (950, 620),
        (980, 720),
        (930, 800),
        (820, 850),
        (700, 870),
        (580, 860),
        (450, 830),
        (330, 800),
        (240, 760),
        (220, 680),
        (250, 600)
    ],


    # --------------------------------------------------------
    # IMAGE 8
    # --------------------------------------------------------
    8: [
        (650, 990),
        (560, 980),
        (470, 970),
        (380, 920),
        (300, 850),
        (270, 760),
        (300, 670),
        (350, 580),
        (400, 500),
        (450, 430),
        (500, 360),
        (540, 280),
        (600, 200),
        (680, 180),
        (730, 250),
        (760, 350),
        (820, 430),
        (900, 440),
        (970, 400),
        (930, 500),
        (850, 560),
        (820, 650),
        (850, 720),
        (950, 780),
        (980, 850),
        (950, 930),
        (850, 980),
        (750, 1000),
        (650, 990)
    ],


    # --------------------------------------------------------
    # IMAGE 9
    # --------------------------------------------------------
    9: [
        (880, 700),
        (900, 780),
        (880, 860),
        (850, 920),
        (780, 960),
        (680, 990),
        (550, 1000),
        (430, 980),
        (330, 930),
        (280, 850),
        (260, 750),
        (250, 650),
        (270, 550),
        (300, 450),
        (350, 370),
        (450, 330),
        (550, 300),
        (650, 260),
        (760, 230),
        (850, 220),
        (930, 260),
        (950, 350),
        (950, 450),
        (940, 550),
        (900, 650),
        (880, 700)
    ],


    # --------------------------------------------------------
    # IMAGE 10
    # --------------------------------------------------------
    10: [
        (330, 830),
        (270, 800),
        (230, 740),
        (200, 650),
        (230, 560),
        (280, 500),
        (350, 460),
        (430, 430),
        (500, 380),
        (520, 300),
        (600, 250),
        (720, 220),
        (840, 220),
        (930, 260),
        (970, 340),
        (950, 450),
        (920, 560),
        (950, 650),
        (1020, 720),
        (1080, 800),
        (1060, 870),
        (980, 900),
        (850, 920),
        (720, 930),
        (650, 900),
        (560, 870),
        (480, 850),
        (400, 830),
        (330, 830)
    ]
}


# ============================================================
# ROAD DETECTION
# ============================================================

def detect_road(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Blur reduces image noise
    blurred = cv2.GaussianBlur(
        gray,
        (9, 9),
        0
    )

    # The road is slightly brighter than the background
    road = (
        blurred > 90
    ).astype(np.uint8)

    # Find connected regions
    number, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(
            road,
            connectivity=8
        )

    # Largest connected region = road
    largest_label = (
        1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    )

    road = (
        labels == largest_label
    ).astype(np.uint8)


    # Close small gaps
    kernel = np.ones(
        (15, 15),
        np.uint8
    )

    road = cv2.morphologyEx(
        road,
        cv2.MORPH_CLOSE,
        kernel
    )


    # --------------------------------------------------------
    # Fill holes inside road
    # --------------------------------------------------------

    flood = (
        road * 255
    ).astype(np.uint8)

    flood_filled = flood.copy()

    flood_mask = np.zeros(
        (
            flood.shape[0] + 2,
            flood.shape[1] + 2
        ),
        np.uint8
    )

    cv2.floodFill(
        flood_filled,
        flood_mask,
        (0, 0),
        255
    )

    holes = cv2.bitwise_not(
        flood_filled
    )

    road = cv2.bitwise_or(
        flood,
        holes
    )

    road = (
        road > 0
    ).astype(np.uint8)


    # --------------------------------------------------------
    # Safety distance from road boundary
    # --------------------------------------------------------

    road = cv2.erode(
        road,
        np.ones((21, 21), np.uint8),
        iterations=1
    )

    return road


# ============================================================
# OBSTACLE + POTHOLE DETECTION
# ============================================================

def detect_obstacles(image, road):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )


    # Color difference
    chromatic_difference = (
        image.max(axis=2)
        -
        image.min(axis=2)
    )


    # --------------------------------------------------------
    # COLORED OBJECTS
    # --------------------------------------------------------

    colored_objects = (
        chromatic_difference > 20
    )


    # --------------------------------------------------------
    # DARK OBJECTS / POTHOLES
    # --------------------------------------------------------

    dark_objects = (
        (gray < 78)
        &
        (hsv[:, :, 2] < 120)
    )


    # Combine both
    obstacle_mask = (
        colored_objects
        |
        dark_objects
    ).astype(np.uint8)


    # Only objects INSIDE the road
    obstacle_mask = (
        obstacle_mask
        &
        road
    )


    # Close small gaps
    obstacle_mask = cv2.morphologyEx(
        obstacle_mask,
        cv2.MORPH_CLOSE,
        np.ones((9, 9), np.uint8)
    )


    # --------------------------------------------------------
    # REMOVE VERY SMALL NOISE
    # --------------------------------------------------------

    number, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(
            obstacle_mask,
            connectivity=8
        )

    clean_mask = np.zeros_like(
        obstacle_mask
    )

    for i in range(1, number):

        area = stats[
            i,
            cv2.CC_STAT_AREA
        ]

        if area > 50:

            clean_mask[
                labels == i
            ] = 1


    # --------------------------------------------------------
    # SAFETY BUFFER
    # --------------------------------------------------------

    # The vehicle/path should not touch the obstacle.
    # Therefore enlarge every obstacle.

    safety_kernel = np.ones(
        (21, 21),
        np.uint8
    )

    obstacle_mask = cv2.dilate(
        clean_mask,
        safety_kernel,
        iterations=1
    )


    return obstacle_mask


# ============================================================
# SNAP POINT TO SAFE ROAD
# ============================================================

def snap_to_mask(point, mask):

    x, y = point

    height, width = mask.shape


    # Already valid
    if (
        0 <= x < width
        and
        0 <= y < height
        and
        mask[y, x] == 1
    ):

        return (
            int(x),
            int(y)
        )


    # Otherwise find nearest valid pixel

    ys, xs = np.where(
        mask == 1
    )

    if len(xs) == 0:
        return point


    distances = (
        (xs - x) ** 2
        +
        (ys - y) ** 2
    )

    index = np.argmin(
        distances
    )

    return (
        int(xs[index]),
        int(ys[index])
    )


# ============================================================
# A* PATH PLANNING
# ============================================================

def a_star(
    free_space,
    start,
    goal,
    scale=4
):

    height, width = free_space.shape


    # Reduce image size for faster A*
    small_width = width // scale
    small_height = height // scale


    small_map = cv2.resize(
        free_space.astype(np.uint8),
        (small_width, small_height),
        interpolation=cv2.INTER_AREA
    )


    small_map = (
        small_map > 0.5
    )


    # Convert original coordinates
    # to reduced coordinates

    start_small = (
        start[0] // scale,
        start[1] // scale
    )

    goal_small = (
        goal[0] // scale,
        goal[1] // scale
    )


    # --------------------------------------------------------
    # Make sure start and goal are free
    # --------------------------------------------------------

    def nearest_free(point):

        x, y = point

        if (
            0 <= x < small_width
            and
            0 <= y < small_height
            and
            small_map[y, x]
        ):
            return point


        ys, xs = np.where(
            small_map
        )

        if len(xs) == 0:
            return point


        distances = (
            (xs - x) ** 2
            +
            (ys - y) ** 2
        )

        index = np.argmin(
            distances
        )

        return (
            int(xs[index]),
            int(ys[index])
        )


    start_small = nearest_free(
        start_small
    )

    goal_small = nearest_free(
        goal_small
    )


    # --------------------------------------------------------
    # Priority queue
    # --------------------------------------------------------

    open_set = []

    heapq.heappush(
        open_set,
        (
            0,
            start_small
        )
    )


    came_from = {
        start_small: None
    }


    cost_so_far = {
        start_small: 0.0
    }


    # 8 possible movements
    directions = [

        (-1, 0, 1.0),
        (1, 0, 1.0),

        (0, -1, 1.0),
        (0, 1, 1.0),

        (-1, -1, 1.414),
        (1, -1, 1.414),

        (-1, 1, 1.414),
        (1, 1, 1.414)
    ]


    # --------------------------------------------------------
    # A* SEARCH
    # --------------------------------------------------------

    while open_set:

        current_priority, current = \
            heapq.heappop(
                open_set
            )


        if current == goal_small:
            break


        for dx, dy, movement_cost in directions:

            next_node = (
                current[0] + dx,
                current[1] + dy
            )


            # Outside image
            if not (
                0 <= next_node[0] < small_width
                and
                0 <= next_node[1] < small_height
            ):
                continue


            # Obstacle or outside road
            if not small_map[
                next_node[1],
                next_node[0]
            ]:
                continue


            new_cost = (
                cost_so_far[current]
                +
                movement_cost
            )


            if (
                next_node not in cost_so_far
                or
                new_cost
                <
                cost_so_far[next_node]
            ):

                cost_so_far[next_node] = new_cost

                # Heuristic
                heuristic = math.hypot(
                    next_node[0] - goal_small[0],
                    next_node[1] - goal_small[1]
                )


                priority = (
                    new_cost
                    +
                    heuristic
                )


                heapq.heappush(
                    open_set,
                    (
                        priority,
                        next_node
                    )
                )


                came_from[
                    next_node
                ] = current


    # --------------------------------------------------------
    # No path found
    # --------------------------------------------------------

    if goal_small not in came_from:

        return []


    # --------------------------------------------------------
    # Reconstruct path
    # --------------------------------------------------------

    path = []

    current = goal_small

    while current is not None:

        x = (
            current[0] * scale
            +
            scale // 2
        )

        y = (
            current[1] * scale
            +
            scale // 2
        )

        path.append(
            (x, y)
        )

        current = came_from[
            current
        ]


    path.reverse()

    return path


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    image_number
):

    filename = f"{image_number}.jpeg"

    input_path = os.path.join(
        INPUT_FOLDER,
        filename
    )


    image = cv2.imread(
        input_path
    )


    if image is None:

        print(
            "Could not read:",
            input_path
        )

        return


    print()
    print(
        "======================================"
    )
    print(
        f"Processing Image {image_number}"
    )
    print(
        "======================================"
    )


    # --------------------------------------------------------
    # 1. Detect road
    # --------------------------------------------------------

    road = detect_road(
        image
    )


    # --------------------------------------------------------
    # 2. Detect obstacles and potholes
    # --------------------------------------------------------

    obstacles = detect_obstacles(
        image,
        road
    )


    # --------------------------------------------------------
    # 3. Create safe/free space
    # --------------------------------------------------------

    free_space = road.copy()

    free_space[
        obstacles > 0
    ] = 0


    # --------------------------------------------------------
    # 4. Get checkpoints
    # --------------------------------------------------------

    checkpoints = CHECKPOINTS[
        image_number
    ]


    # Snap every checkpoint
    # to safe road

    checkpoints = [

        snap_to_mask(
            point,
            road
        )

        for point in checkpoints
    ]


    # --------------------------------------------------------
    # 5. Run A* between every checkpoint
    # --------------------------------------------------------

    complete_path = []

    failed_segments = 0


    for i in range(
        len(checkpoints) - 1
    ):

        start = checkpoints[i]

        goal = checkpoints[i + 1]


        path = a_star(
            free_space,
            start,
            goal,
            scale=4
        )


        if len(path) == 0:

            print(
                f"A* failed between "
                f"checkpoint {i + 1} "
                f"and {i + 2}"
            )

            failed_segments += 1

            continue


        if len(complete_path) == 0:

            complete_path.extend(
                path
            )

        else:

            complete_path.extend(
                path[1:]
            )


    # --------------------------------------------------------
    # 6. Draw output
    # --------------------------------------------------------

    output = image.copy()


    # Draw the detected safe road
    # boundary very lightly

    road_contours, _ = cv2.findContours(
        road.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    cv2.drawContours(
        output,
        road_contours,
        -1,
        (255, 150, 0),
        2
    )


    # --------------------------------------------------------
    # Draw obstacle safety zones
    # --------------------------------------------------------

    obstacle_contours, _ = cv2.findContours(
        obstacles.astype(np.uint8),
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
    # Draw A* path
    # --------------------------------------------------------

    if len(complete_path) > 1:

        path_array = np.array(
            complete_path,
            dtype=np.int32
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
    # Draw checkpoints
    # --------------------------------------------------------

    for i, point in enumerate(
        checkpoints[:-1],
        1
    ):

        x, y = point


        cv2.circle(
            output,
            (x, y),
            5,
            (255, 0, 255),
            -1
        )


        cv2.putText(
            output,
            f"C{i}",
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 255),
            1,
            cv2.LINE_AA
        )


    # --------------------------------------------------------
    # START POINT
    # --------------------------------------------------------

    start_x, start_y = checkpoints[0]


    cv2.circle(
        output,
        (start_x, start_y),
        10,
        (0, 255, 0),
        -1
    )


    cv2.putText(
        output,
        "START",
        (start_x + 12, start_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


    # --------------------------------------------------------
    # INFORMATION ON IMAGE
    # --------------------------------------------------------

    cv2.putText(
        output,
        "A* SAFE PATH",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_filename = (
        f"path_{image_number}.jpeg"
    )


    output_path = os.path.join(
        OUTPUT_FOLDER,
        output_filename
    )


    cv2.imwrite(
        output_path,
        output
    )


    print(
        "Output saved:",
        output_path
    )


    print(
        "A* failed segments:",
        failed_segments
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

print()
print(
    "AERIAL PATH PLANNING"
)

print(
    "Starting detection..."
)


for image_number in range(
    1,
    11
):

    process_image(
        image_number
    )


print()
print(
    "======================================"
)

print(
    "ALL IMAGES PROCESSED"
)

print(
    "Check the 'output' folder."
)

print(
    "Green = Safe A* Path"
)

print(
    "Red = Obstacle Safety Zone"
)

print(
    "Purple = Checkpoints"
)

print(
    "======================================"
)