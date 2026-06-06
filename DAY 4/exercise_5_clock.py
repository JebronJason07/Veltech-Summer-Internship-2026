# exercise_5_clock.py

import numpy as np
import cv2
import math

BLACK = (0, 0, 0)

canvas = np.ones((500, 500, 3), dtype="uint8") * 255

center = (250, 250)
radius = 200

# Draw clock face outline
cv2.circle(canvas, center, radius, BLACK, 5)

# Draw hour markers
for hour in range(12):
    angle = hour * 30
    rad = math.radians(angle - 90)

    inner_x = int(center[0] + (radius - 20) * math.cos(rad))
    inner_y = int(center[1] + (radius - 20) * math.sin(rad))

    outer_x = int(center[0] + radius * math.cos(rad))
    outer_y = int(center[1] + radius * math.sin(rad))

    thickness = 5 if hour % 3 == 0 else 2

    cv2.line(
        canvas,
        (inner_x, inner_y),
        (outer_x, outer_y),
        BLACK,
        thickness
    )

    # Add hour numbers
    if hour != 0:
        num_x = int(center[0] + (radius - 40) * math.cos(rad))
        num_y = int(center[1] + (radius - 40) * math.sin(rad)) + 10

        cv2.putText(
            canvas,
            str(hour),
            (num_x - 10, num_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            BLACK,
            2
        )

# Hour hand (10)
hour_angle = math.radians(300 - 90)
hour_end = (
    int(center[0] + 100 * math.cos(hour_angle)),
    int(center[1] + 100 * math.sin(hour_angle))
)

cv2.line(canvas, center, hour_end, BLACK, 8)

# Minute hand (2)
min_angle = math.radians(60 - 90)
min_end = (
    int(center[0] + 140 * math.cos(min_angle)),
    int(center[1] + 140 * math.sin(min_angle))
)

cv2.line(canvas, center, min_end, BLACK, 4)

# Center dot
cv2.circle(canvas, center, 8, BLACK, -1)

cv2.imshow("Analog Clock", canvas)
cv2.waitKey(0)

cv2.imwrite("clock.png", canvas)

cv2.destroyAllWindows()

print("Clock saved as 'clock.png'")