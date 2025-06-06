import numpy as np
from collections import namedtuple
import mediapipe_utils as mpu
import cv2
from pathlib import Path
from FPS import FPS, now
import argparse
import os
#from openvino.inference_engine import IENetwork, IECore
from openvino.runtime import Core  
from math import *
import base64

import open3d as o3d
from o3d_utils import create_segment, create_grid
import time

SCRIPT_DIR = Path(__file__).resolve().parent
POSE_DETECTION_MODEL = SCRIPT_DIR / "models/pose_detection_FP32.xml"
LANDMARK_MODEL_FULL = SCRIPT_DIR / "models/pose_landmark_full_FP32.xml"
LANDMARK_MODEL_LITE = SCRIPT_DIR / "models/pose_landmark_lite_FP32.xml"
LANDMARK_MODEL_HEAVY = SCRIPT_DIR / "models/pose_landmark_heavy_FP32.xml"

# LINES_*_BODY are used when drawing the skeleton onto the source image. 
# Each variable is a list of continuous lines.
# Each line is a list of keypoints as defined at https://google.github.io/mediapipe/solutions/pose.html#pose-landmark-model-blazepose-ghum-3d
LINES_FULL_BODY = [[28,30,32,28,26,24,12,11,23,25,27,29,31,27], 
                    [23,24],
                    [22,16,18,20,16,14,12], 
                    [21,15,17,19,15,13,11],
                    [8,6,5,4,0,1,2,3,7],
                    [10,9],
                    ]
LINES_UPPER_BODY = [[12,11,23,24,12], 
                    [22,16,18,20,16,14,12], 
                    [21,15,17,19,15,13,11],
                    [8,6,5,4,0,1,2,3,7],
                    [10,9],
                    ]
# LINE_MESH_*_BODY are used when drawing the skeleton in 3D. 
rgb = {"good":(0,255,0), "bad":(0,0,255), "adequate":(0,255,255)}

LINE_MESH_FULL_BODY = [ [9,10],[4,6],[1,3],
                        [12,14],[14,16],[16,20],[20,18],[18,16],
                        [12,11],[11,23],[23,24],[24,12],
                        [11,13],[13,15],[15,19],[19,17],[17,15],
                        [24,26],[26,28],[32,30],
                        [23,25],[25,27],[29,31]]
LINE_TEST = [ [12,11],[11,23],[23,24],[24,12]]

COLORS_FULL_BODY = ["adequate","adequate","adequate","adequate","adequate",
            "adequate","adequate","adequate","adequate","adequate",
            "adequate","adequate","adequate","adequate","adequate",
            "adequate","adequate","adequate","adequate","adequate",
            "adequate","adequate","adequate","adequate","adequate",
            "adequate","adequate","adequate","adequate","adequate",
            "adequate","adequate","adequate"]

COLORS_FULL_BODY = [rgb[x] for x in COLORS_FULL_BODY]

LINE_MESH_UPPER_BODY = [[9,10],[4,6],[1,3],
                        [12,14],[14,16],[16,20],[20,18],[18,16],
                        [12,11],[11,23],[23,24],[24,12],
                        [11,13],[13,15],[15,19],[19,17],[17,15]
                        ]

class BlazeposeOpenvino:
    def __init__(self, input_src=None,
                pd_xml=POSE_DETECTION_MODEL, 
                pd_device="CPU",
                pd_score_thresh=0.5, pd_nms_thresh=0.3,
                lm_xml=LANDMARK_MODEL_FULL,
                lm_device="CPU",
                lm_score_threshold=0.5,
                pose_correction=True,
                smoothing= True,
                filter_window_size=5,
                filter_velocity_scale=10,
                show_3d=False,
                crop=False,
                multi_detection=False,
                force_detection=False,
                output=None,
                thresholds=None
            ):
        self.thresholds = thresholds or {
            "head_angle": {"min": 85, "max": 95},
            "shoulder_angle": {"min": -2.5, "max": 2.5},
            "hips_angle": {"min": -5, "max": 5},
            "head_lean": {"min": -0.3, "max": 0.3},
            "body_lean": {"min": -0.1, "max": 0.1}
        }
        
        self.pd_score_thresh = pd_score_thresh
        self.pd_nms_thresh = pd_nms_thresh
        self.lm_score_threshold = lm_score_threshold
        self.full_body = True
        self.pose_correction = pose_correction
        self.smoothing = smoothing
        self.show_3d = show_3d
        self.crop = crop
        self.multi_detection = multi_detection
        self.force_detection = force_detection
        if self.multi_detection:
            print("Warning: with multi-detection, smoothing filter is disabled and pose detection is forced on every frame.")
            self.smoothing = False
            self.force_detection = True
        
        if input_src.endswith('.jpg') or input_src.endswith('.png') :
            self.input_type= "image"
            self.img = cv2.imread(input_src)
            self.video_fps = 25
            video_height, video_width = self.img.shape[:2]
        else:
            self.input_type = "video"
            if input_src.isdigit():
                input_type = "webcam"
                input_src = int(input_src)
            self.cap = cv2.VideoCapture(input_src)
            self.video_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("Video FPS:", self.video_fps)

        # The full body landmark model predicts 39 landmarks.
        # We are interested in the first 35 landmarks 
        # from 1 to 33 correspond to the well documented body parts,
        # 34th (mid hips) and 35th (a point above the head) are used to predict ROI of next frame
        self.nb_lms = 35

        if self.smoothing:
            self.filter = mpu.LandmarksSmoothingFilter(filter_window_size, filter_velocity_scale, (self.nb_lms-2, 3))
    
        # Load Openvino models
        self.load_models(pd_xml, pd_device, lm_xml, lm_device)

        # Create SSD anchors 
        # https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_detection/pose_detection_cpu.pbtxt

        anchor_options = mpu.SSDAnchorOptions(
                                num_layers=5, 
                                min_scale=0.1484375,
                                max_scale=0.75,
                                input_size_height=224,
                                input_size_width=224,
                                anchor_offset_x=0.5,
                                anchor_offset_y=0.5,
                                strides=[8, 16, 32, 32, 32],
                                aspect_ratios= [1.0],
                                reduce_boxes_in_lowest_layer=False,
                                interpolated_scale_aspect_ratio=1.0,
                                fixed_anchor_size=True)

        self.anchors = mpu.generate_anchors(anchor_options)
        self.nb_anchors = self.anchors.shape[0]
        print(f"{self.nb_anchors} anchors have been created")

        # Rendering flags
        self.show_pd_box = False
        self.show_pd_kps = False
        self.show_rot_rect = False
        self.show_landmarks = True
        self.show_scores = False
        self.show_posture = self.pose_correction
        self.show_fps = False
        self.show_segmentation = False

        if self.show_3d:
            self.vis3d = o3d.visualization.Visualizer()
            self.vis3d.create_window() 
            opt = self.vis3d.get_render_option()
            opt.background_color = np.asarray([0, 0, 0])
            z = min(video_height, video_width)/3
            self.grid_floor = create_grid([0,video_height,-z],[video_width,video_height,-z],[video_width,video_height,z],[0,video_height,z],5,2, color=(1,1,1))
            self.grid_wall = create_grid([0,0,z],[video_width,0,z],[video_width,video_height,z],[0,video_height,z],5,2, color=(1,1,1))
            self.vis3d.add_geometry(self.grid_floor)
            self.vis3d.add_geometry(self.grid_wall)
            view_control = self.vis3d.get_view_control()
            view_control.set_up(np.array([0,-1,0]))
            view_control.set_front(np.array([0,0,-1]))

        if output is None:
            self.output = None
        else:
            if self.input_type == "image":
                # For an source image, we will output one image (and not a video) and exit
                self.output = output
            else:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                self.output = cv2.VideoWriter(output,fourcc,self.video_fps,(video_width, video_height)) 

        # Initialize posture feedback history
        self.posture_feedback_history = []
        self.max_feedback_history = 10  # Store last 10 feedback messages
        self.feedback_display_duration = 3  # Display feedback for 3 seconds
        self.last_feedback_time = time.time()
        
    def load_models(self, pd_xml, pd_device, lm_xml, lm_device):

        print("Loading Inference Engine")
        self.ie = Core()
        print("Device info:")
        versions = self.ie.get_versions(pd_device)
        print("{}{}".format(" "*8, pd_device))
        print("{}MKLDNNPlugin version ......... {}.{}".format(" "*8, versions[pd_device].major, versions[pd_device].minor))
        print("{}Build ........... {}".format(" "*8, versions[pd_device].build_number))

       # Pose detection model
        pd_name = os.path.splitext(pd_xml)[0]
        pd_bin = pd_name + '.bin'
        print("Pose Detection model - Reading network files:\n\t{}\n\t{}".format(pd_xml, pd_bin))
        self.pd_model = self.ie.read_model(model=str(pd_xml), weights=str(pd_bin))
        # In new API, use the model's inputs list.
        self.pd_input_blob = self.pd_model.inputs[0]
        print(f"Input blob: {self.pd_input_blob.get_any_name()} - shape: {self.pd_input_blob.shape}")
        _, _, self.pd_h, self.pd_w = self.pd_input_blob.shape
        for output in self.pd_model.outputs:
            print(f"Output blob: {output.get_any_name()} - shape: {output.shape}")
        self.pd_scores = "Identity_1"
        self.pd_bboxes = "Identity"
        print("Loading pose detection model into the plugin")
        self.pd_exec_net = self.ie.compile_model(model=self.pd_model, device_name=pd_device)
        self.pd_infer_time_cumul = 0
        self.pd_infer_nb = 0

        # Landmarks model
        if lm_device != pd_device:
            print("Device info:")
            versions = self.ie.get_versions(lm_device)
            print(" " * 8, lm_device)
            print(" " * 8, "OpenVINO version: {}.{}".format(versions[lm_device].major, versions[lm_device].minor))
            print(" " * 8, "Build:", versions[lm_device].build_number)

        lm_name = os.path.splitext(lm_xml)[0]
        lm_bin = lm_name + '.bin'
        print("Landmark model - Reading network files:\n\t{}\n\t{}".format(lm_xml, lm_bin))
        self.lm_model = self.ie.read_model(model=str(lm_xml), weights=str(lm_bin))
        self.lm_input_blob = self.lm_model.inputs[0]
        print(f"Input blob: {self.lm_input_blob.get_any_name()} - shape: {self.lm_input_blob.shape}")
        _, _, self.lm_h, self.lm_w = self.lm_input_blob.shape
        for output in self.lm_model.outputs:
            print(f"Output blob: {output.get_any_name()} - shape: {output.shape}")
        self.lm_score = "output_poseflag"
        self.lm_segmentation = "output_segmentation"
        self.lm_landmarks = "ld_3d"
        for output in self.lm_model.outputs:
            if output.get_any_name() == self.lm_segmentation:
                self.segmentation_size = output.shape[-1]
                break
        else:
            raise RuntimeError(f"Output {self.lm_segmentation} not found in landmark model outputs.")
        print("Loading landmark model to the plugin")
        self.lm_exec_net = self.ie.compile_model(model=self.lm_model, device_name=lm_device)
        self.lm_infer_time_cumul = 0
        self.lm_infer_nb = 0

    def pd_postprocess(self, inference):
        scores = np.squeeze(inference[self.pd_scores])  # 2254
        bboxes = inference[self.pd_bboxes][0] # 2254x12
        # Decode bboxes
        self.regions = mpu.decode_bboxes(self.pd_score_thresh, scores, bboxes, self.anchors, best_only=not self.multi_detection)
        # Non maximum suppression (not needed if best_only is True)
        if self.multi_detection: 
            self.regions = mpu.non_max_suppression(self.regions, self.pd_nms_thresh)
        
        mpu.detections_to_rect(self.regions, kp_pair=[0,1] if self.full_body else [2,3])
        mpu.rect_transformation(self.regions, self.frame_size, self.frame_size)

    def pd_render(self, frame):
        for r in self.regions:
            if self.show_pd_box:
                box = (np.array(r.pd_box) * self.frame_size).astype(int)
                cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0,255,0), 2)
            if self.show_pd_kps:
                # Key point 0 - mid hip center
                # Key point 1 - point that encodes size & rotation (for full body)
                # Key point 2 - mid shoulder center
                # Key point 3 - point that encodes size & rotation (for upper body)
                if self.full_body:
                    # Only kp 0 and 1 used
                    list_kps = [0, 1]
                else:
                    # Only kp 2 and 3 used for upper body
                    list_kps = [2, 3]
                for kp in list_kps:
                    x = int(r.pd_kps[kp][0] * self.frame_size)
                    y = int(r.pd_kps[kp][1] * self.frame_size)
                    cv2.circle(frame, (x, y), 3, (0,0,255), -1)
                    cv2.putText(frame, str(kp), (x, y+12), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 2)
            if self.show_scores and r.pd_score is not None:
                cv2.putText(frame, f"Pose score: {r.pd_score:.2f}", 
                        (50, self.frame_size//2), 
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,0), 2)

    def lm_postprocess(self, region, inference):
        region.lm_score = np.squeeze(inference[self.lm_score])
        if region.lm_score > self.lm_score_threshold:  
            self.nb_active_regions += 1

            lm_raw = inference[self.lm_landmarks].reshape(-1,5)
            # Each keypoint have 5 information:
            # - X,Y coordinates are local to the region of
            # interest and range from [0.0, 255.0].
            # - Z coordinate is measured in "image pixels" like
            # the X and Y coordinates and represents the
            # distance relative to the plane of the subject's
            # hips, which is the origin of the Z axis. Negative
            # values are between the hips and the camera;
            # positive values are behind the hips. Z coordinate
            # scale is similar with X, Y scales but has different
            # nature as obtained not via human annotation, by
            # fitting synthetic data (GHUM model) to the 2D
            # annotation.
            # - Visibility, after user-applied sigmoid denotes the
            # probability that a keypoint is located within the
            # frame and not occluded by another bigger body
            # part or another object.
            # - Presence, after user-applied sigmoid denotes the
            # probability that a keypoint is located within the
            # frame.

            # Normalize x,y,z. Here self.lm_w = self.lm_h and scaling in z = scaling in x = 1/self.lm_w
            lm_raw[:,:3] /= self.lm_w
            # Apply sigmoid on visibility and presence (if used later)
            # lm_raw[:,3:5] = 1 / (1 + np.exp(-lm_raw[:,3:5]))
            
            # region.landmarks contains the landmarks normalized 3D coordinates in the relative oriented body bounding box
            region.landmarks = lm_raw[:,:3]
            # Calculate the landmark coordinate in square padded image (region.landmarks_padded)
            src = np.array([(0, 0), (1, 0), (1, 1)], dtype=np.float32)
            dst = np.array([ (x, y) for x,y in region.rect_points[1:]], dtype=np.float32) # region.rect_points[0] is left bottom point and points going clockwise!
            mat = cv2.getAffineTransform(src, dst)
            lm_xy = np.expand_dims(region.landmarks[:self.nb_lms,:2], axis=0)
            lm_xy = np.squeeze(cv2.transform(lm_xy, mat))  
            # A segment of length 1 in the coordinates system of body bounding box takes region.rect_w_a pixels in the
            # original image. Then I arbitrarily divide by 4 for a more realistic appearance.
            lm_z = region.landmarks[:self.nb_lms,2:3] * region.rect_w_a / 4
            lm_xyz = np.hstack((lm_xy, lm_z))
            if self.smoothing:
                lm_xyz = self.filter.apply(lm_xyz)
            region.landmarks_padded = lm_xyz.astype(int)
            # If we added padding to make the image square, we need to remove this padding from landmark coordinates
            # region.landmarks_abs contains absolute landmark coordinates in the original image (padding removed))
            region.landmarks_abs = region.landmarks_padded.copy()
            if self.pad_h > 0:
                region.landmarks_abs[:,1] -= self.pad_h
            if self.pad_w > 0:
                region.landmarks_abs[:,0] -= self.pad_w

            if self.pose_correction: self.posture_classification(region,COLORS_FULL_BODY)

            if self.show_segmentation:
                self.seg = np.squeeze(inference[self.lm_segmentation]) 
                self.seg = 1 / (1 + np.exp(-self.seg))

    def posture_classification(self, r, COLORS_FULL_BODY):      
        posture_status={}
        head_angle_status = "Unknown"
        shoulder_angle_status = "Unknown"
        hips_angle_status = "Unknown"
        head_lean_status = "Unknown"
        body_lean_status = "Unknown"
        feedback={}
        head_angle_feedback=""
        head_angle_feedback = ""
        shoulder_angle_feedback = ""
        hips_angle_feedback = ""
        head_lean_feedback = ""
        body_lean_feedback = ""
        success=0
        overall_posture='Unknown'
        good=(0,255,0)
        bad=(0,0,255)

        # Use thresholds passed to constructor
        thresholds = self.thresholds

        head_orientation=360*atan2(r.landmarks[0,1]-r.landmarks[34,1],r.landmarks[0,0]-r.landmarks[34,0])/2/pi
        shoulder_orientation=360*atan2(r.landmarks[11,1]-r.landmarks[12,1],r.landmarks[11,0]-r.landmarks[12,0])/2/pi
        hips_orientation=360*atan2(r.landmarks[23,1]-r.landmarks[24,1],r.landmarks[23,0]-r.landmarks[24,0])/2/pi
        head_lean=r.landmarks[0,2]-((r.landmarks[11,2]+r.landmarks[12,2])/2)
        body_lean=((r.landmarks[11,2]+r.landmarks[12,2])/2)

        #HEAD
        head_angle=head_orientation-shoulder_orientation
        #print(head_angle)
        if head_angle<thresholds["head_angle"]["min"]:
            head_angle_status = "Head Tilt" ## UPDATE TO TAKE INTO ACCOUNT QUALITY AT END
            head_angle_feedback = "Head is leaning towards the left side." ##CHECK IF NOT CORRECT
            for i in range(11):
                COLORS_FULL_BODY[i]=bad
        elif head_angle>thresholds["head_angle"]["max"]:
            head_angle_status = "Head Tilt" ## UPDATE TO TAKE INTO ACCOUNT QUALITY AT END
            head_angle_feedback = "Head is leaning towards the right side."
            for i in range(11):
                COLORS_FULL_BODY[i]=bad
        else:
            head_angle_status = "Head Well-Positioned"
            head_angle_feedback = 'Well Done!'
            success+=1
            for i in range(11):
                COLORS_FULL_BODY[i]=good
        
        #SHOULDERS
        #print(shoulder_orientation)
        if shoulder_orientation<thresholds["shoulder_angle"]["min"]:
            shoulder_angle_status = "Shoulder Tilt" 
            shoulder_angle_feedback = "Shoulders are leaning towards the left side." 
            for i in range(11,15):
                COLORS_FULL_BODY[i]=bad
        elif shoulder_orientation>thresholds["shoulder_angle"]["max"]:
            shoulder_angle_status = "Shoulder Tilt"
            shoulder_angle_feedback = "Shoulders are leaning towards the right side."
            for i in range(11,15):
                COLORS_FULL_BODY[i]=bad
        else:
            shoulder_angle_status = "Shoulders Well-Positioned"
            shoulder_angle_feedback = 'Well Done!'
            success+=1
            for i in range(11,15):
                COLORS_FULL_BODY[i]=good

        #HIPS
        #print(hips_orientation)
        if hips_orientation<thresholds["hips_angle"]["min"]:
            hips_angle_status = "Hip Tilt" 
            hips_angle_feedback = "Hips are leaning towards the left side. Make sure the hips are far back in the wheelchair and aligned." ##CHECK IF NOT CORRECT
            for i in range(23,27):
                COLORS_FULL_BODY[i]=bad
        elif hips_orientation>thresholds["hips_angle"]["max"]:
            hips_angle_status = "Hip Tilt"
            hips_angle_feedback = "Hips are leaning towards the right side. Make sure the hips are far back in the wheelchair and aligned."
            for i in range(23,27):
                COLORS_FULL_BODY[i]=bad
        else:
            hips_angle_status = "Hips Well-Positioned"
            hips_angle_feedback = 'Well Done!'
            success+=1
            for i in range(23,27):
                COLORS_FULL_BODY[i]=good

        #HEAD LEANING FORWARD
        #print(head_lean)
        if head_lean<thresholds["head_lean"]["min"]:
            head_lean_status = "Neck Lean" 
            head_lean_feedback = "Head is leaning towards the front." 
            for i in range(11):
                COLORS_FULL_BODY[i]=bad
        elif head_lean>thresholds["head_lean"]["max"]:
            head_lean_status = "Neck Lean" 
            head_lean_feedback = "Head is leaning towards the back."
            for i in range(11):
                COLORS_FULL_BODY[i]=bad
        else:
            head_lean_status = "Neck Well-Positioned"
            head_lean_feedback = 'Well Done!'
            success+=1

        #BODY LEANING FORWARD
        #print(body_lean)
        if body_lean<thresholds["body_lean"]["min"]:
            body_lean_status = "Body Lean" 
            body_lean_feedback = "Body is leaning towards the front." 
            for i in range(11,15):
                COLORS_FULL_BODY[i]=bad
        elif body_lean>thresholds["body_lean"]["max"]:
            body_lean_status = "Body Lean" 
            body_lean_feedback = "Body is leaning towards the back."
            for i in range(11,15):
                COLORS_FULL_BODY[i]=bad
        else:
            body_lean_status = "Body Well-Positioned"
            body_lean_feedback = 'Well Done!'
            success+=1 

        posture_quality=success*20

        if posture_quality<50:
            overall_posture='BAD'
        elif posture_quality<75:
            overall_posture='ADEQUATE'
        else:
            overall_posture='GOOD'

        feedback['head_angle_feedback']=head_angle_feedback
        feedback['shoulder_angle_feedback']=shoulder_angle_feedback
        feedback['hips_angle_feedback']=hips_angle_feedback
        feedback['head_lean_feedback']=head_lean_feedback
        feedback['body_lean_feedback']=body_lean_feedback

        posture_status['head_angle_status']=head_angle_status
        posture_status['shoulder_angle_status']=shoulder_angle_status
        posture_status['hips_angle_status']=hips_angle_status
        posture_status['head_lean_status']=head_lean_status
        posture_status['body_lean_status']=body_lean_status

        # Store feedback
        self.add_feedback(feedback, posture_status, overall_posture, posture_quality)
        
        # Set posture display 
        r.posture = posture_status

    def add_feedback(self, feedback, posture_status, overall_posture, posture_quality):
        """Add feedback to history with timestamp"""
        now = time.time()
        self.posture_feedback_history.append({
            'feedback': feedback,
            'posture_status': posture_status,
            'overall_posture': overall_posture,
            'posture_quality': posture_quality,
            'time': now
        })
        
        # Keep only recent feedback
        if len(self.posture_feedback_history) > self.max_feedback_history:
            self.posture_feedback_history.pop(0)
        
        # Update timestamp for display purposes
        self.last_feedback_time = now

    def get_posture_feedback_history(self):
        return self.posture_feedback_history

    def lm_render(self, frame, region):
        if region.lm_score > self.lm_score_threshold:
            if self.show_segmentation:
                ret, mask = cv2.threshold(self.seg, 0.5, 1, cv2.THRESH_BINARY)
                mask = (mask * 255).astype(np.uint8)
                cv2.imshow("seg", self.seg)
                # cv2.imshow("mask", mask)
                src = np.array([[0,0],[self.segmentation_size,0],[self.segmentation_size,self.segmentation_size]], dtype=np.float32) # rect_points[0] is left bottom point !
                dst = np.array(region.rect_points[1:], dtype=np.float32)
                mat = cv2.getAffineTransform(src, dst)
                mask = cv2.warpAffine(mask, mat, (self.frame_size, self.frame_size))
                # cv2.imshow("mask2", mask)
                # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                l = frame.shape[0]
                frame2 = cv2.bitwise_and(frame, frame, mask=mask)
                if not self.crop:
                    frame2 = frame2[self.pad_h:l-self.pad_h, self.pad_w:l-self.pad_w]
                cv2.imshow("Segmentation", frame2)
            if self.show_rot_rect:
                cv2.polylines(frame, [np.array(region.rect_points)], True, (0,255,255), 2, cv2.LINE_AA)
            if self.show_landmarks:
                
                list_connections = LINES_FULL_BODY if self.full_body else LINES_UPPER_BODY
                lines = [np.array([region.landmarks_padded[point,:2] for point in line]) for line in list_connections]
                cv2.polylines(frame, lines, False, (255, 180, 90), 2, cv2.LINE_AA)

                colors = COLORS_FULL_BODY
                
                for i,x_y in enumerate(region.landmarks_padded[:self.nb_lms-2,:2]):
                    color=colors[i]
                    cv2.circle(frame, (x_y[0], x_y[1]), 4, color, -11)

                if self.show_3d:
                    points = region.landmarks_abs
                    lines = LINE_MESH_FULL_BODY if self.full_body else LINE_MESH_UPPER_BODY
                    colors = COLORS_FULL_BODY
                    for i,a_b in enumerate(lines):
                        a, b = a_b
                        line = create_segment(points[a], points[b], radius=5, color=colors[i])
                        if line: self.vis3d.add_geometry(line, reset_bounding_box=False)
                    
            if self.show_scores:
                cv2.putText(frame, f"Landmark score: {region.lm_score:.2f}", 
                        (region.landmarks_padded[24,0]-10, region.landmarks_padded[24,1]+90), 
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,0), 2)
                
    def run(self):

        self.fps = FPS()

        nb_pd_inferences = 0
        nb_pd_inferences_direct = 0 
        nb_lm_inferences = 0
        nb_lm_inferences_after_landmarks_ROI = 0
        glob_pd_rtrip_time = 0
        glob_lm_rtrip_time = 0

        get_new_frame = True
        use_previous_landmarks = False

        global_time = time.perf_counter()
        while True:
            if get_new_frame:
                
                if self.input_type == "image":
                    vid_frame = self.img
                else:
                    ok, vid_frame = self.cap.read()
                    if not ok:
                        break
                h, w = vid_frame.shape[:2]
                if self.crop:
                    # Cropping the long side to get a square shape
                    self.frame_size = min(h, w)
                    dx = (w - self.frame_size) // 2
                    dy = (h - self.frame_size) // 2
                    video_frame = vid_frame[dy:dy+self.frame_size, dx:dx+self.frame_size]
                else:
                    # Padding on the small side to get a square shape
                    self.frame_size = max(h, w)
                    self.pad_h = int((self.frame_size - h)/2)
                    self.pad_w = int((self.frame_size - w)/2)
                    video_frame = cv2.copyMakeBorder(vid_frame, self.pad_h, self.pad_h, self.pad_w, self.pad_w, cv2.BORDER_CONSTANT)
                annotated_frame = video_frame.copy()

            if not self.force_detection and use_previous_landmarks:
                self.regions = regions_from_landmarks
                mpu.detections_to_rect(self.regions, kp_pair=[0,1]) # self.regions.pd_kps are initialized from landmarks on previous frame
                mpu.rect_transformation(self.regions, self.frame_size, self.frame_size)
            else:
                # Infer pose detection
                # Resize image to NN square input shape
                frame_nn = cv2.resize(video_frame, (self.pd_w, self.pd_h), interpolation=cv2.INTER_AREA)
                # Transpose hxwx3 -> 1x3xhxw
                frame_nn = np.transpose(frame_nn, (2,0,1))[None,]
    
                pd_rtrip_time = now()
                inference = self.pd_exec_net({self.pd_input_blob.get_any_name(): frame_nn})
                glob_pd_rtrip_time += now() - pd_rtrip_time
                self.pd_postprocess(inference)
                self.pd_render(annotated_frame)
                nb_pd_inferences += 1
                if get_new_frame: nb_pd_inferences_direct += 1


            # Landmarks
            self.nb_active_regions = 0
            if self.show_3d:
                self.vis3d.clear_geometries()
                self.vis3d.add_geometry(self.grid_floor, reset_bounding_box=False)
                self.vis3d.add_geometry(self.grid_wall, reset_bounding_box=False)
            if self.force_detection:
                for r in self.regions:
                    frame_nn = mpu.warp_rect_img(r.rect_points, video_frame, self.lm_w, self.lm_h)
                    # Transpose hxwx3 -> 1x3xhxw
                    frame_nn = np.transpose(frame_nn, (2,0,1))[None,] / 255.0
                    # Get landmarks
                    lm_rtrip_time = now()
                    inference = self.lm_exec_net({self.lm_input_blob.get_any_name(): frame_nn})
                    glob_lm_rtrip_time += now() - lm_rtrip_time
                    nb_lm_inferences += 1
                    self.lm_postprocess(r, inference)
                    self.lm_render(annotated_frame, r)
            elif len(self.regions) == 1:
                r = self.regions[0]
                frame_nn = mpu.warp_rect_img(r.rect_points, video_frame, self.lm_w, self.lm_h)
                # Transpose hxwx3 -> 1x3xhxw
                frame_nn = np.transpose(frame_nn, (2,0,1))[None,] / 255.0
                # Get landmarks
                lm_rtrip_time = now()
                inference = self.lm_exec_net({self.lm_input_blob.get_any_name(): frame_nn})
                glob_lm_rtrip_time += now() - lm_rtrip_time
                nb_lm_inferences += 1
                if use_previous_landmarks:
                    nb_lm_inferences_after_landmarks_ROI += 1

                self.lm_postprocess(r, inference)
                if not self.force_detection:
                    if get_new_frame:
                        if not use_previous_landmarks: 
                            # With a new frame, we have run the landmark NN on a ROI found by the detection NN...  
                            if r.lm_score > self.lm_score_threshold:
                                # ...and succesfully found a body and its landmarks
                                # Predict the ROI for the next frame from the last 2 landmarks normalized coordinates (x,y)
                                regions_from_landmarks = [mpu.Region(pd_kps=r.landmarks_padded[self.nb_lms-2:self.nb_lms,:2]/self.frame_size)]
                                use_previous_landmarks = True
                        else : 
                            # With a new frame, we have run the landmark NN on a ROI calculated from the landmarks of the previous frame...
                            if r.lm_score > self.lm_score_threshold:
                                # ...and succesfully found a body and its landmarks
                                # Predict the ROI for the next frame from the last 2 landmarks normalized coordinates (x,y)
                                regions_from_landmarks = [mpu.Region(pd_kps=r.landmarks_padded[self.nb_lms-2:self.nb_lms,:2]/self.frame_size)]
                                use_previous_landmarks = True
                            else:
                                # ...and could not find a body
                                # We don't know if it is because the ROI calculated from the previous frame is not reliable (the body moved)
                                # or because there is really no body in the frame. To decide, we have to run the detection NN on this frame
                                get_new_frame = False
                                use_previous_landmarks = False 
                                continue
                    else:
                        # On a frame on which we already ran the landmark NN without founding a body,
                        # we have run the detection NN...
                        if r.lm_score > self.lm_score_threshold:
                            # ...and succesfully found a body and its landmarks
                            use_previous_landmarks = True
                            # Predict the ROI for the next frame from the last 2 landmarks normalized coordinates (x,y)
                            regions_from_landmarks = [mpu.Region(pd_kps=r.landmarks_padded[self.nb_lms-2:self.nb_lms,:2]/self.frame_size)]
                            use_previous_landmarks = True
                        # else:
                            # ...and could not find a body
                            # We are sure there is no body in that frame
                        
                        get_new_frame = True
                self.lm_render(annotated_frame, r) 
            else:
                # Detection NN hasn't found any body
                get_new_frame = True

            self.fps.update()  
                                 
            if self.show_3d:
                self.vis3d.poll_events()
                self.vis3d.update_renderer()
            if self.smoothing and self.nb_active_regions == 0:
                self.filter.reset()

            if not self.crop:
                annotated_frame = annotated_frame[self.pad_h:self.pad_h+h, self.pad_w:self.pad_w+w]

            if self.show_fps:
                self.fps.draw(annotated_frame, orig=(50,50), size=1, color=(240,180,100))
            #cv2.imshow("Blazepose", annotated_frame)
            imgencode = cv2.imencode('.jpg', annotated_frame)[1]

            if self.output:
                if self.input_type == "image":
                    cv2.imwrite(self.output, annotated_frame)
                    break
                else:
                    self.output.write(annotated_frame)

            key = cv2.waitKey(1) 
            if key == ord('q') or key == 27:
                break
            elif key == 32:
                # Pause on space bar
                cv2.waitKey(0)
            elif key == ord('1'):
                self.show_pd_box = not self.show_pd_box
            elif key == ord('2'):
                self.show_pd_kps = not self.show_pd_kps
            elif key == ord('3'):
                self.show_rot_rect = not self.show_rot_rect
            elif key == ord('4'):
                self.show_landmarks = not self.show_landmarks
            elif key == ord('5'):
                self.show_scores = not self.show_scores
            elif key == ord('6'):
                self.show_posture = not self.show_posture
            elif key == ord('f'):
                self.show_fps = not self.show_fps
            elif key == ord('s'):
                self.show_segmentation = not self.show_segmentation

        # Print some stats
        if nb_pd_inferences > 1:
            global_fps, nb_frames = self.fps.get_global()
            print(f"FPS : {global_fps:.1f} f/s (# frames = {nb_frames})")
            print(f"# pose detection inferences : {nb_pd_inferences} - # direct: {nb_pd_inferences_direct} - # after landmarks ROI failures: {nb_pd_inferences-nb_pd_inferences_direct}")
            print(f"# landmark inferences       : {nb_lm_inferences} - # after pose detection: {nb_lm_inferences - nb_lm_inferences_after_landmarks_ROI} - # after landmarks ROI prediction: {nb_lm_inferences_after_landmarks_ROI}")
            print(f"Pose detection round trip   : {glob_pd_rtrip_time/nb_pd_inferences*1000:.1f} ms")
            if nb_lm_inferences: 
                print(f"Landmark round trip         : {glob_lm_rtrip_time/nb_lm_inferences*1000:.1f} ms")

        if self.output and self.input_type != "image":
            self.output.release()

        return base64.b64encode(imgencode).decode('utf-8')
           
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='0', 
                        help="Path to video or image file to use as input (default=%(default)s)")
    parser.add_argument('-g', '--posture', action="store_true", 
                        help="enable posture recognition")
    parser.add_argument("--pd_xml", type=str,
                        help="Path to an .xml file for pose detection model")
    parser.add_argument("--pd_device", default='CPU', type=str,
                        help="Target device for the pose detection model (default=%(default)s)")  
    parser.add_argument("--lm_xml", type=str,
                        help="Path to an .xml file for landmark model")
    parser.add_argument("--lm_version", type=str, choices=['full', 'lite', 'heavy'], default="full",
                        help="Version of the landmark model (default=%(default)s)")
    parser.add_argument("--lm_device", default='CPU', type=str,
                        help="Target device for the landmark regression model (default=%(default)s)")
    parser.add_argument('--min_tracking_conf', type=float, default=0.5,
                        help="Minimum confidence value ([0.0, 1.0]) from the landmark-tracking model for the pose landmarks to be considered tracked successfully,"+
                        " or otherwise person detection will be invoked automatically on the next input image. (default=%(default)s)")                    
    parser.add_argument('-c', '--crop', action="store_true", 
                        help="Center crop frames to a square shape before feeding pose detection model")
    parser.add_argument('--no_smoothing', action="store_true", 
                        help="Disable smoothing filter")
    parser.add_argument('--filter_window_size', type=int, default=5,
                        help="Smoothing filter window size. Higher value adds to lag and to stability (default=%(default)i)")                    
    parser.add_argument('--filter_velocity_scale', type=float, default=10,
                        help="Smoothing filter velocity scale. Lower value adds to lag and to stability (default=%(default)s)")                    
    parser.add_argument('-3', '--show_3d', action="store_true", 
                        help="Display skeleton in 3d in a separate window (valid only for full body landmark model)")
    parser.add_argument("-o","--output",
                        help="Path to output video file")
    parser.add_argument('--multi_detection', action="store_true", 
                        help="Force multiple person detection (at your own risk, the original Mediapipe implementation is designed for one person tracking)")
    parser.add_argument('--force_detection', action="store_true", 
                        help="Force person detection on every frame (never use landmarks from previous frame to determine ROI)")

    args = parser.parse_args()

    if not args.pd_xml:
        args.pd_xml = POSE_DETECTION_MODEL
    if not args.lm_xml:
        if args.lm_version == "full":
            args.lm_xml = LANDMARK_MODEL_FULL
        elif args.lm_version == "lite":
            args.lm_xml = LANDMARK_MODEL_LITE
        elif args.lm_version == "heavy":
            args.lm_xml = LANDMARK_MODEL_HEAVY
    ht = BlazeposeOpenvino(input_src=args.input, 
                    pd_xml=args.pd_xml,
                    pd_device=args.pd_device, 
                    lm_xml=args.lm_xml,
                    lm_device=args.lm_device,
                    lm_score_threshold=args.min_tracking_conf,
                    smoothing=not args.no_smoothing,
                    filter_window_size=args.filter_window_size,
                    filter_velocity_scale=args.filter_velocity_scale,
                    pose_correction=args.posture,
                    show_3d=args.show_3d,
                    crop=args.crop,
                    multi_detection=args.multi_detection,
                    force_detection=args.force_detection,
                    output=args.output)
    ht.run()
