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
    SEARCHING = 1
    ONE = 2
    ALIGN = 3
    GRAB = 4

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
        self.inc = 0.05
        self.inc1 = 0.1
        self.inc2 = -0.1
        self.inc3 = 0.1
        self.inc4 = 0.001
        self.last_time = time.time()
        self.centered_tag = None
        self.tagX = 320
        self.tagY = 250
        self.last_time = None
        # Global state of the progress of the arm during the catching sequence
        # (No tags found, one tag found, both tags found)
        self.state = STATES.NONE

        return


    def run(self,
            image: list,
            global_poses: Dict[str, np.ndarray],
            calcIK: Callable[[np.ndarray, Optional[np.ndarray]], Dict[str, float]]
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
            tag_centers.append( ((cX, cY), r.tag_id))
            cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
            # draw the tag family on the image
            tagFamily = r.tag_family.decode("utf-8")
            cv2.putText(image, tagFamily, (ptA[0], ptA[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(image, str(r.tag_id), (ptA[0], ptA[1] - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 4)
            print("[INFO] tag family: {}".format(tagFamily))

        # --------------- END OF COPIED CODE -----------------
        # Update states

        if len(tag_centers) == 0 and self.state == STATES.ONE:
            self.state = STATES.NONE

        if len(tag_centers) > 0 and self.state == STATES.SEARCHING:
            self.state = STATES.ONE

        if self.state == STATES.NONE:
            #self.pose = calcIK(np.array([3, 0, 1.6]), np.array([1, 0, -1, 0]))
            self.state = STATES.SEARCHING

        elif self.state == STATES.SEARCHING:
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
            if self.pose['bravo_axis_g'] > 1.5 * math.pi:
                self.inc4 = -0.1
            if self.pose['bravo_axis_g'] < 0.5 * math.pi:
                self.inc4 = 0.1
            self.pose["bravo_axis_c"] += self.inc1
            self.pose["bravo_axis_e"] += self.inc2
            self.pose["bravo_axis_f"] += self.inc3
            self.pose["bravo_axis_g"] -= self.inc4



        elif self.state == STATES.ONE:
            ## Zoom out if centered
            if len(tag_centers) == 2:
                first_tag_dist = math.sqrt( (320-tag_centers[0][0][0])**2 + (250-tag_centers[0][0][1])**2 )
                second_tag_dist = math.sqrt( (320-tag_centers[1][0][0])**2 + (250-tag_centers[1][0][1])**2 )

                if first_tag_dist < second_tag_dist:
                    self.tagX, self.tagY = tag_centers[0][0]
                    tag_id = tag_centers[0][1]
                else:
                    self.tagX, self.tagY = tag_centers[1][0]
                    tag_id = tag_centers[1][1]
            else:
                self.tagX, self.tagY = tag_centers[0][0]
                tag_id = tag_centers[0][1]

            # Center the April Tag
            if (self.tagX >= 310 and self.tagX <= 330) and (self.tagY >= 240 and self.tagY <= 260):
                # Centered
                self.centered_tag = tag_id
                # reorient hand to point correctly
#                fl = 640 /(2 * math.tan(100 * math.pi / 360))
#
#                intrinsicMatrix = np.asmatrix([[fl, 0, 320],[0, fl, 240],[0,0,1]])
#                tagPos = np.matmul(np.linalg.inv(intrinsicMatrix), np.asmatrix([[self.tagX], [self.tagY], [1]])) * cam_pos[2]
#
#                if self.centered_tag == 0:
#                   current_pos = (current_pos[0] + tagPos[0] + 0.25, current_pos[1] +tagPos[1] - 0.12, current_pos[2] +tagPos[2])
#                if self.centered_tag == 1:
#                   current_pos = (current_pos[0] + tagPos[0] - 0.2, current_pos[1] +tagPos[1] - 0.12, current_pos[2] +tagPos[2])
                self.pose = calcIK(current_pos, np.array([1,0,-1,0]))
                self.state = STATES.ALIGN

            if self.tagX < 310:
                self.pose["bravo_axis_e"] -= 0.025
            if self.tagX > 330:
                self.pose["bravo_axis_e"] += 0.025
            if self.tagY < 240:
                self.pose["bravo_axis_g"] -=  0.025
            if self.tagY > 260:
                self.pose["bravo_axis_g"] += 0.025


        elif self.state == STATES.ALIGN:
                if self.last_time == None:
                    self.last_time = time.time()
                    if self.centered_tag == 0:
                       current_pos += np.array([0.25, -0.06, 0])
                    if self.centered_tag == 1:
                       current_pos += np.array([-0.1, -0.12, 0])
                    self.pose = calcIK(current_pos, current_quat)

                elif time.time() - self.last_time >= 0.5:
                    self.last_time = None
                    self.state = STATES.GRAB


        elif self.state == STATES.GRAB:
            if current_pos[2] > 0.25:
                self.inc = 0
                self.state = STATES.NONE

            current_pos += np.array([0, 0, self.inc - 0.1])
            self.pose = calcIK(current_pos, current_quat)
            self.inc += 0.005


        # EXAMPLE USAGE OF INVERSE KINEMATICS SOLVER
        #   Inputs: vec3 position, quaternion orientation
        # self.pose = calcIK(np.array([0.8, 0, 0.4]), np.array([1, 0, 0, 0]))

        cv2.putText(image, str(self.state) + f" TAG: {self.centered_tag}", (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("View", image)
        cv2.waitKey(1)
        return self.pose
