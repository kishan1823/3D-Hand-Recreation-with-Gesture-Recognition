import cv2
import torch
from manopth import manolayer
from model.detnet import detnet
from model import KeyPointClassifier
from utils import func, bone, AIK, smoother ,CvFpsCalc
import numpy as np
import matplotlib.pyplot as plt
from utils import vis
from op_pso import PSO
import open3d
import mediapipe as mp
import csv
import copy
import argparse
import itertools
from collections import deque
def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    _mano_root = 'mano/models'

    module = detnet().to(device)
    print('load model start')
    check_point = torch.load('checkpoints/bmc_ckp.pth', map_location=device)
    model_state = module.state_dict()
    state = {}
    for k, v in check_point.items():
        if k in model_state:
            state[k] = v
        else:
            print(k, ' is NOT in current model')
    model_state.update(state)
    module.load_state_dict(model_state)
    print('load model finished')
    pose, shape = func.initiate("zero")
    pre_useful_bone_len = np.zeros((1, 15))
    pose0 = torch.eye(3).repeat(1, 16, 1, 1)

    mano = manolayer.ManoLayer(flat_hand_mean=True,
                            side="right",
                            mano_root=_mano_root,
                            use_pca=False,
                            root_rot_mode='rotmat',
                            joint_rot_mode='rotmat')
    print('start opencv')
    point_fliter = smoother.OneEuroFilter(4.0, 0.0)
    mesh_fliter = smoother.OneEuroFilter(4.0, 0.0)
    shape_fliter = smoother.OneEuroFilter(4.0, 0.0)
    cap = cv2.VideoCapture(0)
    print('opencv finished')
    flag = 1
    plt.ion()
    f = plt.figure(figsize=(6,6))
    fliter_ax = f.add_subplot(111, projection='3d')
    plt.show()
    plt.show(block=False)
    view_mat = np.array([[1.0, 0.0, 0.0],
                        [0.0, -1.0, 0],
                        [0.0, 0, -1.0]])
    mesh = open3d.geometry.TriangleMesh()
    hand_verts, j3d_recon = mano(pose0, shape.float())
    mesh.triangles = open3d.utility.Vector3iVector(mano.th_faces)
    hand_verts = hand_verts.clone().detach().cpu().numpy()[0]
    mesh.vertices = open3d.utility.Vector3dVector(hand_verts)
    viewer = open3d.visualization.Visualizer()
    viewer.create_window(width=480, height=480, window_name='mesh')
    viewer.add_geometry(mesh)
    viewer.update_renderer()

    print('start pose estimate')

    pre_uv = None
    shape_time = 0
    opt_shape = None
    shape_flag = True
    mp_drawing = mp.solutions.drawing_utils # used for drawing keypoints on opencv window
    mp_hands = mp.solutions.hands # used to extract keyoints

    keypoint_classifier = KeyPointClassifier()
    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
                encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    #  ########################################################################
    mode = 0
    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        while (cap.isOpened()):
            fps = cvFpsCalc.get()
            key = cv2.waitKey(10)
            if key == 27:  # ESC
                break
            number, mode = select_mode(key, mode)
            ret_flag, img = cap.read()
            image1 = cv2.flip(img, 1)  # Mirror display
            debug_image = copy.deepcopy(image1)
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

            image1.flags.writeable = False
            results = hands.process(image1)
            image1.flags.writeable = True
            input = np.flip(img.copy(), -1)
            k = cv2.waitKey(1) & 0xFF
            if input.shape[0] > input.shape[1]:
                margin = (input.shape[0] - input.shape[1]) // 2
                input = input[margin:-margin]
            else:
                margin = (input.shape[1] - input.shape[0]) // 2
                input = input[:, margin:-margin]
            img = input.copy()
            img = np.flip(img, -1)
            image=img
            #cv2.imshow("Capture_Test", img)
            input = cv2.resize(input, (128, 128))
            input = torch.tensor(input.transpose([2, 0, 1]), dtype=torch.float, device=device) 
            input = func.normalize(input, [0.5, 0.5, 0.], [1, 1, 1])
            result = module(input.unsqueeze(0))

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image.flags.writeable = False
            results1 = hands.process(image)
            s=np.zeros((21,3))  # declaring 21X3 null matrix for keypoints
            results=hands.process(image1)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results1.multi_hand_landmarks:
            
                for hand_landmarks in results1.multi_hand_landmarks:
                    
                    # saving (x,y,z) axis keypoints of hand 
                    for i in range(21):
                        s[i][0]= hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x 
                        s[i][1]= hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y 
                        s[i][2]= hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z 

            if results.multi_hand_landmarks:
                
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                    results.multi_handedness):
                    
                    
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = pre_process_landmark(
                        landmark_list)

                    # Write to the dataset file
                    logging_csv(number, mode, pre_processed_landmark_list
                                )

                    # Hand sign classification
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                    point_history.append([0, 0])

                    # Drawing part
                    
                    debug_image = draw_landmarks(debug_image, landmark_list)
                    debug_image = draw_info_text(
                        debug_image,
                        handedness,
                        keypoint_classifier_labels[hand_sign_id],    
                    )
            else:
                point_history.append([0, 0])

            debug_image = draw_point_history(debug_image, point_history)
            debug_image = draw_info(debug_image, fps, mode, number)

            # Screen reflection #############################################################
            cv2.imshow('Hand Gesture Recognition', debug_image)
            
            
            
            pre_joints = result['xyz'].squeeze(0)
            now_uv = result['uv'].clone().detach().cpu().numpy()[0, 0]
            now_uv = now_uv.astype(np.float)
            trans = np.zeros((1, 3))
            trans[0, 0:2] = now_uv - 16.0
            trans = trans / 16.0
            #new_tran = np.array([[trans[0, 1], trans[0, 0], trans[0, 2]]])
            new_tran=s
            pre_joints = pre_joints.clone().detach().cpu().numpy()
            

            flited_joints = point_fliter.process(pre_joints)

            fliter_ax.cla()

            filted_ax = vis.plot3d(flited_joints + new_tran, fliter_ax)
            pre_useful_bone_len = bone.caculate_length(pre_joints, label="useful")

            NGEN = 100
            popsize = 100
            low = np.zeros((1, 10)) - 3.0
            up = np.zeros((1, 10)) + 3.0
            parameters = [NGEN, popsize, low, up]
            pso = PSO(parameters, pre_useful_bone_len.reshape((1, 15)))
            pso.main()
            opt_shape = pso.ng_best
            opt_shape = shape_fliter.process(opt_shape)
            opt_tensor_shape = torch.tensor(opt_shape, dtype=torch.float)
            _, j3d_p0_ops = mano(pose0, opt_tensor_shape)
            template = j3d_p0_ops.cpu().numpy().squeeze(0) / 1000.0 
            ratio = np.linalg.norm(template[9] - template[0]) / np.linalg.norm(pre_joints[9] - pre_joints[0])
            j3d_pre_process = pre_joints * ratio 
            j3d_pre_process = j3d_pre_process - j3d_pre_process[0] + template[0]
            pose_R = AIK.adaptive_IK(template, j3d_pre_process)
            pose_R = torch.from_numpy(pose_R).float()
            #  reconstruction
            hand_verts, j3d_recon = mano(pose_R, opt_tensor_shape.float())
            mesh.triangles = open3d.utility.Vector3iVector(mano.th_faces)
            hand_verts = hand_verts.clone().detach().cpu().numpy()[0]
            hand_verts = mesh_fliter.process(hand_verts)
            hand_verts = np.matmul(view_mat, hand_verts.T).T
            hand_verts[:, 0] = hand_verts[:, 0] - 50
            hand_verts[:, 1] = hand_verts[:, 1] - 50
            mesh_tran = np.array([[-new_tran[0, 0], new_tran[0, 1], new_tran[0, 2]]])
            hand_verts = hand_verts + mesh_tran

            mesh.vertices = open3d.utility.Vector3dVector(hand_verts)
            mesh.paint_uniform_color([228 / 255, 178 / 255, 148 / 255])
            mesh.compute_triangle_normals()
            mesh.compute_vertex_normals()
            viewer.update_geometry(mesh)
            viewer.poll_events()
            if k == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    return


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 手首1
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image

def draw_info_text(image,  handedness, hand_sign_text
                   ):

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
        cv2.putText(image, info_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, info_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv2.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv2.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info(image, fps, mode, number):
    cv2.putText(image, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv2.LINE_AA)

    mode_string = ['Logging Key Point']
    if 1 == mode:
        cv2.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv2.LINE_AA)
        if 0 <= number <= 9:
            cv2.putText(image, "NUM:" + str(number), (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv2.LINE_AA)
    return image

if __name__ == '__main__':
    main()
