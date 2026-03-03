from datetime import date, datetime
import cv2
import numpy as np
import os


img_pts = np.empty((0, 2), dtype=np.float64)

# Mouse callback function
def get_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Collected pixel: {x},{y}")
        global img_pts
        img_pts = np.append(img_pts, np.array([[x, y]]), axis=0)
        
def calibrate():
    # Load image
    project_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(project_dir, '..', 'calibration/img', 'cap.jpg')
    config_path = os.path.normpath(config_path)
    print(f"Loading image from: {config_path}")

    img = cv2.imread(config_path)  # <-- replace with your image path

    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", get_coordinates)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(img_pts)
    if len(img_pts) < 4:
        print("At least 4 points are required for homography.")
        return
    
    input_str = input("Input corresponding robot coordinates for the collected pixels (format: X1,Y1;X2,Y2;...): ")
    try:
        robot_pts = np.empty((0, 2), dtype=np.float64)
        for pair in input_str.split(';'):
            X_input, Y_input = map(float, pair.split(','))
            
            robot_pts = np.vstack((robot_pts, [X_input, Y_input]))

        print(f"Input robot coordinates: {robot_pts}")

        # Compute homography from image -> robot plane
        H, mask = cv2.findHomography(img_pts, robot_pts, method=0)

        print("Homography matrix H:")
        print(H)

        json_path = os.path.join(project_dir, '..', 'calibration', 'calibration.json')

        with open(f"{json_path}", "w") as f:
            f.write(f"{H}")
            # optional metadata: image size, date, camera id, notes
            f.write(f"\nImage size: {img.shape[1]}x{img.shape[0]}", )
            f.write(f"\nDate: {datetime.now()}", )
    except ValueError:
        
        print("Invalid input format. Please enter coordinates as 'X1,Y1;X2,Y2;...'.")
        return

if __name__ == "__main__":
    calibrate()