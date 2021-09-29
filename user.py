""" User code lives here """
import time
from typing import Dict
import math
from typing import Callable, Optional
import numpy as np
import cv2
import apriltag
import logging
from enum import Enum
logging.basicConfig(filename='user.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

class STATES(Enum):
    NONE = 0
    ONE = 1
    TWO = 2

class User:
    def __init__(self) -> None:
        self.pose = {
            "bravo_axis_a": 0.0,
            "bravo_axis_b": math.pi * 0.5,
            "bravo_axis_c": math.pi * 0.5,
            "bravo_axis_d": math.pi * 0,
            "bravo_axis_e": math.pi * 0.75,
            "bravo_axis_f": math.pi * 0.9,
            "bravo_axis_g": math.pi
        }
        self.inc = 0.1
        self.last_time = time.time()
        # Global state of the progress of the arm during the catching sequence
        # (No tags found, one tag found, both tags found)
        self.state = STATES.NONE
        self.handleX, self.handleY = (320, 240)

        return


    def run(self,
            image: list,
            global_poses: Dict[str, np.ndarray],
            calcIK: Callable[[np.ndarray, Optional[np.ndarray]], Dict[str, float]],
            ) -> Dict[str, float]:
        """Run loop to control the Bravo manipulator.

        Parameters
        ----------
        image: list
            The latest camera image frame.

        global_poses: Dict[str, np.ndarray]
            A dictionary with the global camera and end-effector pose. The keys are
            'camera_end_joint' and 'end_effector_joint'. Each pose consitst of a (3x1)
            position (vec3) and (4x1) quaternion defining the orientation.

        calcIK: function, (pos: np.ndarray, orient: np.ndarray = None) -> Dict[str, float]
            Function to calculate inverse kinematics. Provide a desired end-effector
            position (vec3) and an orientation (quaternion) and it will return a pose
            dictionary of joint angles to approximate the pose.
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        tag_centers = []

        # ---------- COPIED FROM https://www.pyimagesearch.com/2020/11/02/apriltag-with-python/ -----------
        #
        # define the AprilTags detector options and then detect the AprilTags
        # in the input image
        print("[INFO] detecting AprilTags...")
        options = apriltag.DetectorOptions(families="tag36h11")
        detector = apriltag.Detector(options)
        results = detector.detect(gray)
        print("[INFO] {} total AprilTags detected".format(len(results)))
        # loop over the AprilTag detection results
        for r in results:
            # extract the bounding box (x, y)-coordinates for the AprilTag
            # and convert each of the (x, y)-coordinate pairs to integers
            (ptA, ptB, ptC, ptD) = r.corners
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))
            ptA = (int(ptA[0]), int(ptA[1]))
            # draw the bounding box of the AprilTag detection
            cv2.line(image, ptA, ptB, (0, 255, 0), 2)
            cv2.line(image, ptB, ptC, (0, 255, 0), 2)
            cv2.line(image, ptC, ptD, (0, 255, 0), 2)
            cv2.line(image, ptD, ptA, (0, 255, 0), 2)
            # draw the center (x, y)-coordinates of the AprilTag
            (cX, cY) = (int(r.center[0]), int(r.center[1]))
            tag_centers.append( (cX, cY) )
            cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
            # draw the tag family on the image
            tagFamily = r.tag_family.decode("utf-8")
            cv2.putText(image, tagFamily, (ptA[0], ptA[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print("[INFO] tag family: {}".format(tagFamily))

        # --------------- END OF COPIED CODE -----------------
        current_pos, current_quat = global_poses['end_effector_joint']
        # Update states
        if len(tag_centers) == 0 and self.state != STATES.TWO:
            self.state = STATES.NONE

        if len(tag_centers) == 1 and self.state != STATES.TWO:
            self.state = STATES.ONE

        if len(tag_centers) == 2:
            self.state = STATES.TWO


        if self.state == STATES.NONE:
            # If no tags are visible, go to a default position (TODO: needs refinement)
            self.pose = calcIK(np.array([0.8, 0, 1.6]), np.array([1, 0, -1, 0]))

        if self.state == STATES.ONE:
            # Only found one tag so far, center it first
            tagX, tagY = tag_centers[0]
            if tagX < 320:
                self.pose["bravo_axis_e"] -= self.inc
            if tagX > 320:
                self.pose["bravo_axis_e"] += self.inc
            if tagY < 240:
                self.pose["bravo_axis_f"] -=  self.inc
            if tagY > 240:
                self.pose["bravo_axis_f"] += self.inc
            # Zoom out once centered (TODO: figure out quaternion transformation to zoom out at any orientation)
            #if (tagX == 320 and tagY == 240):
            #    current_pos = (current_pos[0], current_pos[1], current_pos[2] + self.inc)
            #    self.pose = calcIK(current_pos, current_quat)



        if self.state == STATES.TWO:
            cv2.circle(image, (self.handleX, self.handleY), 5, (255, 0, 0), 2)
            # Center the handle bar
            if len(tag_centers) == 2:
                self.handleX, self.handleY = (int((tag_centers[0][0] + tag_centers[1][0]) / 2), int((tag_centers[0][1] + tag_centers[1][1]) / 2))
            if self.handleX < 320:
                self.pose["bravo_axis_e"] -= self.inc
            if self.handleX > 320:
                self.pose["bravo_axis_e"] += self.inc
            if self.handleY < 240:
                self.pose["bravo_axis_f"] -=  self.inc
            if self.handleY > 240:
                self.pose["bravo_axis_f"] += self.inc
            # Zoom in
            if (self.handleX == 320 and self.handleY == 240):
                if current_pos[3] > 0:
                    current_pos = (current_pos[0], current_pos[1], current_pos[2] - self.inc)
                    self.pose = calcIK(current_pos, current_quat)

        # THIS IS AN EXAMPLE TO SHOW YOU HOW TO MOVE THE MANIPULATOR

        # EXAMPLE USAGE OF INVERSE KINEMATICS SOLVER
        #   Inputs: vec3 position, quaternion orientation
        # self.pose = calcIK(np.array([0.8, 0, 0.4]), np.array([1, 0, 0, 0]))

        cv2.imshow("View", image)
        cv2.waitKey(1)
        return self.pose
