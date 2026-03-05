import sys
import argparse
import calibration.compute as calibration
import numpy as np
import cv2
import matplotlib.pyplot as plt

from perception.segmentation import detect 
from robot.transform import transform, robot_pick


def main():
    parser = argparse.ArgumentParser(
                    prog='CLI',
                    description='entry point for the machine vision project')
    
    
    subparsers = parser.add_subparsers(dest='command', help='subcommand help')
    subparsers.required = True

    # Calibrate command
    calibrate_parser = subparsers.add_parser(
        'calibrate',
        help='Perform camera calibration'
    )
    calibrate_parser.set_defaults(func='calibrate')
    
    # Detect command
    detect_parser = subparsers.add_parser(
        'detect',
        help='Detect objects and report positions'
    )
    detect_parser.add_argument(
        '--mode',
        choices=['plan', 'execute'],
        required=True,
        help='Operation mode: plan (dry run) or execute (send to robot)'
    )
    detect_parser.add_argument(
        '--color',
        choices=['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'white', 'black'],
        help='Filter by color'
    )
    detect_parser.add_argument(
        '--shape',
        choices=['circle', 'square', 'rectangle', 'triangle', 'hexagon'],
        help='Filter by shape'
    )
    detect_parser.set_defaults(func='detect')
    
    # Pick command
    pick_parser = subparsers.add_parser(
        'pick',
        help='Pick objects based on criteria'
    )
    pick_parser.add_argument(
        '--mode',
        choices=['plan', 'execute'],
        required=True,
        help='Operation mode: plan (dry run) or execute (send to robot)'
    )
    pick_parser.add_argument(
        '--color',
        choices=['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'white', 'black'],
        help='Target color to pick'
    )
    pick_parser.add_argument(
        '--shape',
        choices=['circle', 'square', 'rectangle', 'triangle', 'hexagon'],
        help='Target shape to pick'
    )
    pick_parser.set_defaults(func='pick')
    
    # Parse arguments
    args = parser.parse_args()
    try:
        if args.func == 'calibrate':
            print("cli.calibrate(args)")
            calibration.calibrate()

        elif args.func == 'detect' or args.func == 'pick':
            print("cli.detect(args)", args.mode, args.color, args.shape)

            with open("calibration/calibration.json", "r") as f:
                lines = [f.readline().strip() for _ in range(3)]
                # Parse the array
                array_str = '\n'.join(lines)
                array_str = array_str.replace('[', '').replace(']', '')
                rows = [line.split() for line in array_str.split('\n')]
                H = np.array(rows, dtype=float)

                print(f"Loaded homography matrix H:\n{H}")
                print(f"\nMetadata:\n{f.readline()}\n{f.readline()}")
            

            img = cv2.imread("calibration/img/obj1.jpg")
            obj_pos, img_vis = detect(img, color=args.color, shape=args.shape)

            if len(obj_pos) == 0:
                print("======= No target found. =======")
                return
            else:
                print(f"======= {len(obj_pos)} targets found. =======")
            print(f"Detected object positions in pixel coordinates: {len(obj_pos)} objs. {obj_pos}")
            obj_pos_robot = transform(obj_pos, H)


            if args.func == 'pick':
                if args.mode == 'execute':
                    print("Executing pick (sending to robot)...")
                    robot_pick(obj_pos_robot)
                    print("CLI pick completed.")

    except KeyboardInterrupt:
        print("\n\n  Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()