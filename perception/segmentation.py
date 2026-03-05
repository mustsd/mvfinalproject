import cv2
import numpy as np
import matplotlib.pyplot as plt

# color HSV range
COLOR_RANGES = {
    "yellow": ([20, 80, 80],   [35, 255, 255]),
    "red":    ([0, 100, 100],  [10, 255, 255]),   # red has two ranges in HSV
    "red2":   ([160, 100, 100],[180, 255, 255]),
    "blue":   ([100, 80, 80],  [135, 255, 255]),
    "green":  ([35, 80, 80],   [85, 255, 255]),
    "orange": ([10, 100, 100], [20, 255, 255]),
    "pink":   ([150, 30, 150], [180, 150, 255]), 
    "purple": ([130, 50, 50],  [160, 255, 255]),
    "brown":  ([8, 60, 40],    [20, 200, 180]),   # wooden box color
    "white":  ([0, 0, 180],    [180, 30, 255]),
    "black":  ([0, 0, 0],      [180, 255, 50]),
}

def detect_color(img, color: str):
    """Detect specific color in the image and return a binary mask."""

    img_blur = cv2.GaussianBlur(img, (5,5), 0)

    hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    
    if color not in COLOR_RANGES:
        return np.zeros(img.shape[:2], dtype=np.uint8)
    
    lower, upper = COLOR_RANGES[color]
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    
    # red need to combine two masks
    if color == "red":
        lower2, upper2 = COLOR_RANGES["red2"]
        mask2 = cv2.inRange(hsv, np.array(lower2), np.array(upper2))
        mask = cv2.bitwise_or(mask, mask2)

    # print(f"[{color}] raw mask white pixels: {np.sum(mask == 255)}")

    if color == "blue":
        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=1)
    else:
        kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # print(f"[{color}] after morphology white pixels: {np.sum(mask == 255)}")

    return mask

def segment(img, color=None):
    """Segment the image to extract objects based on color criteria."""
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_stretched = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX)
    img_adjusted = cv2.convertScaleAbs(img_stretched, alpha=1.5, beta=10)

    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(img_adjusted, cv2.MORPH_OPEN, kernel, iterations=1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    T_otsu, thotsu = cv2.threshold(closing, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if color is None:
        # combine mask
        color_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for c in COLOR_RANGES:
            if c in ("red2", "white", "black"):  # skip red2 since it's combined in red, white and black are too broad
                continue
            color_mask = cv2.bitwise_or(color_mask, detect_color(img, c))
        mask = cv2.bitwise_or(thotsu, color_mask)
    else:
        mask = detect_color(img, color)

    return img, gray_img, closing, mask

def detect_shape(contour):
    """Detect shape of the contour and return shape name and approx vertices."""
    shape = "unknown"
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    
    vertices = len(approx)
    if vertices == 3:
        shape = "triangle"
    elif vertices == 4:
        # square or rectangle
        (x, y, w, h) = cv2.boundingRect(approx)
        ratio = w / float(h)
        shape = "square" if 0.78 <= ratio <= 1.3 else "rectangle"
    elif vertices == 5:
        shape = "pentagon"
    else:
        # 顶点多 → 圆形
        area = cv2.contourArea(contour)
        circularity = 4 * np.pi * area / (peri ** 2)
        shape = "circle" if circularity > 0.7 else "unknown"
    
    return shape, approx

def detect_by_shape(seg_img, target_shape: str = None):
    """Detect objects in the segmented image based on shape criteria."""
    contours, _ = cv2.findContours(seg_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # # for debugging: 
    # print(f"seg_img white pixels: {np.sum(seg_img == 255)}")
    # print(f"total contours: {len(contours)}")
    # for i, cnt in enumerate(contours):
    #     area = cv2.contourArea(cnt)
    #     peri = cv2.arcLength(cnt, True)
    #     approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    #     circularity = 4 * np.pi * area / (peri ** 2)
    #     shape, _ = detect_shape(cnt)
    #     print(f"  contour {i}: area={area:.0f}, vertices={len(approx)}, circularity={circularity:.3f}, shape={shape}")

    results = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 500:
            continue
        shape, approx = detect_shape(cnt)
        # calculate object centroid
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        if target_shape is not None and shape != target_shape:
            continue

        results.append({"shape": shape, "cx": cx, "cy": cy})

        # 圆形用外接圆画，其他用 approx 轮廓
        # if shape == "circle":
        #     (x, y), radius = cv2.minEnclosingCircle(cnt)
        #     cv2.circle(img_vis, (int(x), int(y)), int(radius), (0, 0, 255), 3)
        # else:
        #     cv2.drawContours(img_vis, [approx], -1, (0, 0, 255), 2)

    # print(results)
    return results

def detect(img, color=None, shape=None):
    """Detect objects in the image based on color and shape criteria."""
    img, gray_img, closing, mask = segment(img, color=color)  
    img_vis = img.copy()

    results = detect_by_shape(mask, target_shape=shape)
    obj_pos = [[r["cx"], r["cy"]] for r in results]

    # draw centroid and annotation for each detected object
    for (cx, cy) in obj_pos:
        cv2.putText(img_vis, f'({cx},{cy})', (cx + 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(img_vis, (cx, cy), 5, (0, 0, 255), -1)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.imshow(closing, cmap='gray')
    plt.title('Processing')
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.imshow(mask, cmap='gray')
    plt.title('segmentation & filtering')
    plt.axis('off')

    plt.subplot(2, 2, 4)       
    plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
    plt.title('Detection Result')

    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return obj_pos, img_vis
