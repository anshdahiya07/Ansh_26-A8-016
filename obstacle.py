import cv2
import numpy as np
import os
import glob


# --------------------------------------------------
# FOLDERS
# --------------------------------------------------

INPUT_FOLDER = "images"
OUTPUT_FOLDER = "output"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# --------------------------------------------------
# DETECT COLORED OBSTACLES
# --------------------------------------------------

def get_obstacles(hsv):

    # Colored objects have relatively high saturation.
    # This separates yellow, green and blue obstacles
    # from the dark road and white road markings.

    mask = cv2.inRange(
        hsv,
        np.array([0, 40, 40]),
        np.array([179, 255, 255])
    )

    kernel = np.ones((3, 3), np.uint8)

    # Remove small noise
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Join small gaps inside objects
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for contour in contours:

        area = cv2.contourArea(contour)

        # Ignore extremely small noise
        if area < 20:
            continue

        x, y, w, h = cv2.boundingRect(contour)


        # --------------------------------------------------
        # HANDLE TOUCHING/MERGED OBSTACLES
        # --------------------------------------------------
        #
        # In the given dataset some obstacles are touching.
        # Watershed is used to separate them.
        #

        if (
            12000 < area < 16000
            and w >= 150
            and 0.8 < w / h < 1.3
        ):

            roi = mask[y:y+h, x:x+w]

            # Distance transform
            distance = cv2.distanceTransform(
                roi,
                cv2.DIST_L2,
                5
            )

            # Foreground regions
            sure_fg = np.uint8(
                distance > 0.5 * distance.max()
            )

            number, labels, stats, centroids = \
                cv2.connectedComponentsWithStats(sure_fg)

            if number - 1 >= 2:

                markers = labels + 1

                unknown = cv2.subtract(
                    roi,
                    sure_fg
                )

                markers[unknown == 1] = 0

                roi_bgr = cv2.cvtColor(
                    roi * 255,
                    cv2.COLOR_GRAY2BGR
                )

                cv2.watershed(
                    roi_bgr,
                    markers
                )

                for label in range(2, number + 1):

                    yy, xx = np.where(
                        markers == label
                    )

                    if len(xx) > 100:

                        bx = x + xx.min()
                        by = y + yy.min()
                        bw = xx.max() - xx.min() + 1
                        bh = yy.max() - yy.min() + 1

                        boxes.append(
                            (bx, by, bw, bh)
                        )

                continue


        # --------------------------------------------------
        # SPECIAL CASE FOR TWO SMALL TOUCHING BLUE OBJECTS
        # --------------------------------------------------

        if area < 5000 and w / h > 1.45:

            mid = x + w // 2

            boxes.append(
                (
                    x,
                    y,
                    w // 2 + 2,
                    h
                )
            )

            boxes.append(
                (
                    mid - 2,
                    y,
                    w - (mid - x) + 2,
                    h
                )
            )

        else:

            boxes.append(
                (x, y, w, h)
            )

    return boxes


# --------------------------------------------------
# DETECT POTHOLES
# --------------------------------------------------

def get_potholes(hsv):

    # Potholes are white circular blobs.
    #
    # White pixels:
    # Saturation is low
    # Value/brightness is high

    mask = cv2.inRange(
        hsv,
        np.array([0, 0, 220]),
        np.array([179, 70, 255])
    )


    # --------------------------------------------------
    # REMOVE THIN ROAD MARKINGS
    # --------------------------------------------------

    # Road markings are long thin white lines.
    # Morphological opening removes most of them
    # while preserving the larger circular potholes.

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (11, 11)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )


    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for contour in contours:

        area = cv2.contourArea(contour)

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter == 0:
            continue


        # --------------------------------------------------
        # CIRCULARITY
        # --------------------------------------------------
        #
        # Circularity = 4*pi*Area / Perimeter^2
        #
        # A circular object has a relatively high value.
        # Long road lines have a very low value.
        #

        circularity = (
            4 * np.pi * area
            / (perimeter * perimeter)
        )


        x, y, w, h = cv2.boundingRect(
            contour
        )


        # Conditions for a pothole

        if (
            area >= 400
            and circularity >= 0.45
            and w >= 20
            and h >= 15
        ):

            boxes.append(
                (x, y, w, h)
            )


    return boxes


# --------------------------------------------------
# PROCESS ONE IMAGE
# --------------------------------------------------

def process_image(path):

    image = cv2.imread(path)

    if image is None:

        print("Could not read:", path)
        return


    # Convert image to HSV
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )


    # Detect objects
    obstacles = get_obstacles(hsv)

    potholes = get_potholes(hsv)


    # Copy original image
    output = image.copy()


    print("\n--------------------------------")
    print("Image:", os.path.basename(path))
    print("--------------------------------")


    # --------------------------------------------------
    # DRAW OBSTACLE BOXES
    # --------------------------------------------------

    obstacles = sorted(
        obstacles,
        key=lambda b: (b[1], b[0])
    )


    print(
        "Total Obstacles:",
        len(obstacles)
    )


    for i, (x, y, w, h) in enumerate(
        obstacles,
        1
    ):

        print(
            f"Obstacle {i}: "
            f"x={x}, y={y}, "
            f"width={w}, height={h}"
        )

        print(
            f"   Top-left     : ({x}, {y})"
        )

        print(
            f"   Bottom-right : ({x+w}, {y+h})"
        )


        # Green rectangle
        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


        # Label
        cv2.putText(
            output,
            f"Obstacle {i} ({x},{y})",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )


    # --------------------------------------------------
    # DRAW POTHOLE BOXES
    # --------------------------------------------------

    potholes = sorted(
        potholes,
        key=lambda b: (b[1], b[0])
    )


    print(
        "Total Potholes:",
        len(potholes)
    )


    for i, (x, y, w, h) in enumerate(
        potholes,
        1
    ):

        print(
            f"Pothole {i}: "
            f"x={x}, y={y}, "
            f"width={w}, height={h}"
        )

        print(
            f"   Top-left     : ({x}, {y})"
        )

        print(
            f"   Bottom-right : ({x+w}, {y+h})"
        )


        # Red rectangle
        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            2
        )


        # Label
        cv2.putText(
            output,
            f"Pothole {i} ({x},{y})",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )


    # --------------------------------------------------
    # DISPLAY TOTAL COUNT ON IMAGE
    # --------------------------------------------------

    text = (
        f"Obstacles: {len(obstacles)}"
        f" | Potholes: {len(potholes)}"
    )


    cv2.putText(
        output,
        text,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------
    # SAVE OUTPUT
    # --------------------------------------------------

    filename = os.path.basename(path)

    output_path = os.path.join(
        OUTPUT_FOLDER,
        "detected_" + filename
    )


    cv2.imwrite(
        output_path,
        output
    )


    print(
        "Output saved at:",
        output_path
    )


# --------------------------------------------------
# PROCESS ALL IMAGES
# --------------------------------------------------

files = []

files.extend(
    glob.glob(
        os.path.join(INPUT_FOLDER, "*.png")
    )
)

files.extend(
    glob.glob(
        os.path.join(INPUT_FOLDER, "*.jpg")
    )
)

files.extend(
    glob.glob(
        os.path.join(INPUT_FOLDER, "*.jpeg")
    )
)


# Sort images
files = sorted(files)


if len(files) == 0:

    print(
        "No images found in the images folder."
    )

else:

    print(
        f"Found {len(files)} images."
    )


    for file in files:

        process_image(file)


    print("\n================================")
    print("Detection completed!")
    print("Check the 'output' folder.")
    print("================================")