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
import random

logging.basicConfig(filename='user.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

class STATES(Enum):
    NONE = 0
    ONE = 1
    TWO = 2
    SEARCHING = 3

class User:
    def __init__(self) -> None:
        self.pose = {
            "bravo_axis_a": 0,
            "bravo_axis_b": math.pi * 0.5,
            "bravo_axis_c": math.pi * 0.5,
            "bravo_axis_d": math.pi * 0,
            "bravo_axis_e": math.pi * 0.75,
            "bravo_axis_f": math.pi * 0.9,
            "bravo_axis_g": math.pi
        }
        self.inc = 0.05
        self.inc1 = 0.1
        self.inc2 = -0.1
        self.inc3 = 0.1
        self.inc4 = 0.1
        self.inc5 = 0.1
        self.last_time = time.time()
        self.random_flag = True
        # Global state of the progress of the arm during the catching sequence
        # (No tags found, one tag found, both tags found)
        self.state = STATES.NONE
        self.handleX, self.handleY = (320, 240)
        self.adjustments = [0,0]

        return


    def run(self,
            image: list,
            global_poses: Dict[str, np.ndarray],
            calcIK: Callable[[np.ndarray, Optional[np.ndarray]], Dict[str, float]],
            projectionMatrix
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
        current_pos, current_quat = global_poses['end_effector_joint']
        cam_pos, cam_quat = global_poses['camera_end_joint']

        cv2.putText(image, "Effector Position: {0:.4f}, {1:.4f}, {2:.4f}".format(current_pos[0], current_pos[1], current_pos[2]), (10, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, "Camera Position: {0:.4f}, {1:.4f}, {2:.4f}".format(cam_pos[0], cam_pos[1], cam_pos[2]), (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, "Effector Quaternion: {0:.4f}, {1:.4f}, {2:.4f}, {3:.4f}".format(current_quat[0], current_quat[1], current_quat[2], current_quat[3]), (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, "Camera Quaternion: {0:.4f}, {1:.4f}, {2:.4f}, {3:.4f}".format(cam_quat[0], cam_quat[1], cam_quat[2], cam_quat[3]), (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

        # ---------- COPIED FROM https://www.pyimagesearch.com/2020/11/02/apriltag-with-python/ -----------
        #
        # define the AprilTags detector options and then detect theGeometry and Transformations


        # in the input image
        print("[INFO] detecting AprilTags...")
        options = apriltag.DetectorOptions(families="tag36h11")
        detector = apriltag.Detector(options)
        results = detector.detect(gray)
        #print("[INFO] {} total AprilTags detected".format(len(results)))
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
            cv2.putText(image, str(r.tag_id), (ptA[0], ptA[1] - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 4)
            #print("[INFO] tag family: {}".format(tagFamily))

        # --------------- END OF COPIED CODE -----------------
        # Update states

        if self.state == STATES.NONE:
            self.state = STATES.SEARCHING

        if len(tag_centers) == 0 and self.state != STATES.TWO and self.state != STATES.SEARCHING:
            self.state = STATES.NONE

        if len(tag_centers) == 1 and self.state != STATES.TWO:
            self.state = STATES.ONE

        if len(tag_centers) == 2:
            self.state = STATES.TWO


        if self.state == STATES.NONE:
            # If no tags are visible, go to a default position
            self.pose = calcIK(np.array([3, 0, 1.6]), np.array([1, 0, -1, 0]))

        if self.state == STATES.SEARCHING:
            # Go searching for new tags
            if self.pose['bravo_axis_c'] > 1.000 * math.pi :    # +0.500
                self.inc1 = -0.1
            if self.pose['bravo_axis_c'] < 0.250 * math.pi:     # -0.250
                self.inc1 = 0.1
            if self.pose['bravo_axis_e'] > 1.000 * math.pi :    # +0.250
                self.inc2 = -0.1
            if self.pose['bravo_axis_e'] < 0.250 * math.pi:     # -0.500
                self.inc2 = 0.1
            if self.pose['bravo_axis_f'] > 1.400 * math.pi:     # +0.500
                self.inc3 = -0.1
            if self.pose['bravo_axis_f'] < 0.650 * math.pi:     # -0.250
                self.inc3 = 0.1
            if self.pose['bravo_axis_g'] > 1.2 * math.pi:
                self.inc4 = -0.05
            if self.pose['bravo_axis_g'] < 0.80 * math.pi:
                self.inc4 = 0.05
            # if self.pose['bravo_axis_b'] > 0.8 * math.pi:
            #     self.inc5 = -0.05
            # if self.pose['bravo_axis_b'] < -0.8 * math.pi:
            #     self.inc5 = 0.05
            self.pose["bravo_axis_c"] += self.inc1
            self.pose["bravo_axis_e"] += self.inc2
            self.pose["bravo_axis_f"] += self.inc3
            self.pose["bravo_axis_g"] += self.inc4
            # self.pose["bravo_axis_b"] += self.inc5


        if self.state == STATES.ONE:
            ## Zoom out if centered

            tagX, tagY = tag_centers[0]

            if (tagX > 300 and tagX < 340) and (tagY > 220 and tagY < 260):
                #just changes the z out (precalculated)
                current_pos = (current_pos[0], current_pos[1], current_pos[2] + 0.13)
                self.pose = calcIK(current_pos, current_quat)
                # if it is the left one it would extend out
                if r.tag_id == "0":
                    self.pose["bravo_axis_f"] += 0.2
                    self.pose["bravo_axis_c"] -= 0.1
                #if its the other on2e then pull in
                if r.tag_id == "1":
                    self.pose["bravo_axis_f"] -= 0.2
                    self.pose["bravo_axis_e"] += 0.2
                    self.pose["bravo_axis_c"] -= 0.2
            # Center the April Tag
            if tagX < 310:
                self.pose["bravo_axis_e"] -= 0.01
            if tagX > 330:
                self.pose["bravo_axis_e"] += 0.01
            if tagY < 240:
                self.pose["bravo_axis_g"] -=  0.01
            if tagY > 260:
                self.pose["bravo_axis_g"] += 0.01



        if self.state == STATES.TWO:
            cv2.circle(image, (self.handleX, self.handleY), 5, (255, 0, 0), 8)

            if current_pos[2] > 0.4:
                self.random_flag = True
                self.state = STATES.NONE
            #current_pos = (current_pos[0], current_pos[1], current_pos[2] - 0.1 + self.inc)
            self.pose = calcIK(current_pos, current_quat)
            if current_pos[2] < 0 and self.random_flag:
                #self.pose = calcIK(current_pos, current_quat)
                #self.pose["bravo_axis_e"] += self.adjustments[0]
                #self.pose["bravo_axis_g"] -= self.adjustments[1]

                self.random_flag = False
            #if not self.random_flag:
            #           self.pose["bravo_axis_g"] += 0.005 * math.pi

            self.inc += 0.005
            if len(tag_centers) == 2:
                self.inc = 0
                # Get handle bar coordinates when both tags are visible
                self.handleX, self.handleY = (int((tag_centers[0][0] + tag_centers[1][0]) / 2), int((tag_centers[0][1] + tag_centers[1][1]) / 2))
                # Center the handle bar
                # self.pose["bravo_axis_a"] = 0.0
                # self.pose["bravo_axis_b"] = math.pi * 0.5
                # self.pose["bravo_axis_c"] = math.pi * 0.5
                # self.pose["bravo_axis_d"] = math.pi * 0
                # self.pose["bravo_axis_f"] = math.pi * 0.9

                fl = 640 /(2 * math.tan(100 * math.pi / 360))
                intrinsicMatrix = np.asmatrix([[fl, 0, 320],[0, fl, 240],[0,0,1]])
                handlePos = np.matmul(np.linalg.inv(intrinsicMatrix), np.asmatrix([[self.handleX], [self.handleY], [1]])) * cam_pos[2]
                self.pose = calcIK(handlePos, current_quat)

                # This 1.5 constant is tripping us up
                # self.pose["bravo_axis_e"] += (self.handleX-320)/640 * (2.0) * math.pi
                # self.pose["bravo_axis_g"] += (self.handleY-240)/480 * (2.0-(0.25-current_pos[2])/0.25) * math.pi




        # EXAMPLE USAGE OF INVERSE KINEMATICS SOLVER
        #   Inputs: vec3 position, quaternion orientation
        # self.pose = calcIK(np.array([0.8, 0, 0.4]), np.array([1, 0, 0, 0]))

        cv2.putText(image, str(self.state), (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("View", image)
        cv2.waitKey(1)
        return self.pose
