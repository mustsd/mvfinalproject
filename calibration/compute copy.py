from datetime import date, datetime
import cv2
import numpy as np


img_points = np.empty((0, 2), dtype=np.float64)

def detect():

    # Example lists (replace with your data)
    img_pts = np.array([
        [90, 87],
        [532, 83],
        [88, 335],
        [538,332],
        [213,259],
        [357,207],
        [471,269],
        [315,67]
    ], dtype=np.float64)

    robot_pts = np.array([
        [106,-382],
        [-118,-385],
        [104,-257],
        [-121,-261],
        [43,-297],
        [-29,-323],
        [-87,-292],
        [-8,-394]
    ], dtype=np.float64)

    # Compute homography from image -> robot plane
    H, mask = cv2.findHomography(img_pts, robot_pts, method=0)

    print("Homography matrix H:")
    print(H)

    def pixel_to_robot(u, v, H):
        # Input: u, v as floats, H as 3x3 homography
        p = np.array([u, v, 1.0], dtype=np.float32).reshape(3, 1)
        pr = H @ p
        pr = pr / pr[2, 0]  # divide by last coordinate to normalize
        X = pr[0, 0]
        Y = pr[1, 0]
        return X, Y

    # Mouse callback function
    def get_coordinates(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Collected pixel: {x},{y}")
            global img_points
            img_points = np.append(img_points, np.array([[x, y]]), axis=0)
            
    # Load image
    img = cv2.imread("img/calib_image1 copy.jpg")  # <-- replace with your image path

    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", get_coordinates)


    with open("calibration.json", "w") as f:
        f.write(f"{H}")

        # optional metadata: image size, date, camera id, notes
        f.write(f"\nImage size: {img.shape[1]}x{img.shape[0]}", )
        f.write(f"\nDate: {datetime.now()}", )


    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(img_points)
    if len(img_points) < 4:
        print("At least 4 points are required for homography.")
        return
    
    input_str = input("Input corresponding robot coordinates for the collected pixels (format: X1,Y1;X2,Y2;...): ")
    try:
        robot_points = np.empty((0, 2), dtype=np.float64)
        for pair in input_str.split(';'):
            X_input, Y_input = map(float, pair.split(','))
            
            robot_points = np.vstack((robot_points, [X_input, Y_input]))

        print(f"Input robot coordinates: {robot_points}")

    except ValueError:
        
        print("Invalid input format. Please enter coordinates as 'X1,Y1;X2,Y2;...'.")
        return

if __name__ == "__main__":
    detect()