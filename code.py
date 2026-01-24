import cv2
import numpy as np
import random

# Black canvas
img = np.zeros((500, 500, 3), dtype=np.uint8)

# Function to draw random shape
def draw_random_shape(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        shape_type = random.randint(0,4)  # 0-circle,1-rectangle,2-ellipse,3-polygon

        if shape_type == 0:
            radius = random.randint(10,60)
            cv2.circle(img, (x,y), radius, color, -1)
        elif shape_type == 1:
            w,h = random.randint(20,80), random.randint(20,80)
            cv2.rectangle(img, (x-w//2, y-h//2), (x+w//2, y+h//2), color, -1)
        elif shape_type == 2:
            axes = (random.randint(10,60), random.randint(10,60))
            angle = random.randint(0,360)
            cv2.ellipse(img, (x,y), axes, angle, 0, 360, color, -1)
        elif shape_type == 3:
            # Random polygon with 3-6 points
            points = []
            for _ in range(random.randint(3,6)):
                px = x + random.randint(-50,50)
                py = y + random.randint(-50,50)
                points.append([px,py])
            points = np.array([points], np.int32)
            cv2.fillPoly(img, points, color)

# Setup window and callback
cv2.namedWindow("Infinite Shape Generator")
cv2.setMouseCallback("Infinite Shape Generator", draw_random_shape)

# Display loop
while True:
    cv2.imshow("Infinite Shape Generator", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
