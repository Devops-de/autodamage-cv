"""
Generates clean, realistic synthetic car panel test images with scratches and dents.
"""
import os
import cv2
import numpy as np


def generate_samples(output_dir: str = "app/static/sample_images"):
    os.makedirs(output_dir, exist_ok=True)

    w, h = 800, 600

    # 1. Sample 1: Red Metallic Car Bumper with deep key scratches
    bumper = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        grad = y / h
        r = int(170 + grad * 45)
        g = int(15 + grad * 15)
        b = int(20 + grad * 10)
        bumper[y, :] = [b, g, r]

    # Add realistic multi-stroke scratches (white/gray exposed primer lines)
    pts1 = np.array([[220, 180], [310, 240], [450, 270], [560, 310]], np.int32)
    cv2.polylines(bumper, [pts1], isClosed=False, color=(225, 230, 235), thickness=4)

    pts2 = np.array([[240, 215], [330, 260], [420, 285], [510, 315]], np.int32)
    cv2.polylines(bumper, [pts2], isClosed=False, color=(180, 185, 190), thickness=3)

    pts3 = np.array([[380, 310], [460, 340], [520, 355]], np.int32)
    cv2.polylines(bumper, [pts3], isClosed=False, color=(210, 215, 220), thickness=2)

    bumper = cv2.GaussianBlur(bumper, (3, 3), 0.5)
    cv2.imwrite(os.path.join(output_dir, "sample_bumper_scratch.jpg"), bumper)

    # 2. Sample 2: Silver Metallic Door with Concave Dent & Shadow
    door = np.zeros((h, w, 3), dtype=np.uint8)
    for x in range(w):
        val = int(185 + (x / w) * 40)
        door[:, x] = [val, val, val]

    dent_center = (400, 300)
    cv2.ellipse(door, dent_center, (120, 80), 20, 0, 360, (100, 100, 105), -1)
    cv2.ellipse(door, (dent_center[0] - 15, dent_center[1] - 10), (105, 65), 20, 180, 360, (240, 245, 250), 3)
    door = cv2.GaussianBlur(door, (13, 13), 4.0)

    scrape_pts = np.array([[330, 280], [380, 295], [440, 310]], np.int32)
    cv2.polylines(door, [scrape_pts], isClosed=False, color=(60, 60, 65), thickness=4)
    cv2.imwrite(os.path.join(output_dir, "sample_door_dent.jpg"), door)

    # 3. Sample 3: Blue Pearl Fender with Scrape and Paint Chip
    fender = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        b = int(140 + (y / h) * 60)
        g = int(50 + (y / h) * 20)
        r = int(20 + (y / h) * 15)
        fender[y, :] = [b, g, r]

    cv2.circle(fender, (380, 260), 45, (45, 45, 50), -1)
    cv2.circle(fender, (430, 290), 30, (55, 55, 60), -1)
    cv2.ellipse(fender, (360, 240), (25, 15), 45, 0, 360, (180, 180, 190), 2)
    fender = cv2.GaussianBlur(fender, (5, 5), 1.0)
    cv2.imwrite(os.path.join(output_dir, "sample_fender_chip.jpg"), fender)

    print("Regenerated clean test samples successfully!")


if __name__ == "__main__":
    generate_samples()
