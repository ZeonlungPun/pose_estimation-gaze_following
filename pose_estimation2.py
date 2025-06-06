import cv2
import dlib
import numpy as np
import math


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
POINTS_NUM_LANDMARK = 68


# face detection
def _largest_face(dets):
    if len(dets) == 1:
        return 0

    face_areas = [(det.right() - det.left()) * (det.bottom() - det.top()) for det in dets]

    largest_area = face_areas[0]
    largest_index = 0
    for index in range(1, len(dets)):
        if face_areas[index] > largest_area:
            largest_index = index
            largest_area = face_areas[index]

    print("largest_face index is {} in {} faces".format(largest_index, len(dets)))

    return largest_index


def get_head_pose(shape, frame):

    image_points = np.array([
        (shape.part(30).x, shape.part(30).y),
        (shape.part(8).x, shape.part(8).y),
        (shape.part(36).x, shape.part(36).y),
        (shape.part(45).x, shape.part(45).y),
        (shape.part(48).x, shape.part(48).y),
        (shape.part(54).x, shape.part(54).y)
    ], dtype="double")


    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ])

    # camera intrstic matrix
    focal_length = frame.shape[1]
    center = (frame.shape[1] / 2, frame.shape[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    # assume that no distortion of camera len
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs,
                                                                flags=cv2.SOLVEPNP_ITERATIVE)

    if success:
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                         translation_vector, camera_matrix, dist_coeffs)

        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        # 计算欧拉角
        rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
        proj_matrix = np.hstack((rvec_matrix, translation_vector))
        eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

        pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]
        pitch = -math.degrees(pitch)
        yaw = -math.degrees(yaw)
        roll = math.degrees(roll)

        return pitch, yaw, roll, p1, p2
    else:
        return None, None, None, None, None


def determine_head_pose_state(pitch, yaw, roll):
    yaw_orientation,pitch_orientation, roll_orientation= "","", ""
    if abs(yaw) < 15:
        yaw_orientation += "Facing Front"
    elif yaw < -15:
        yaw_orientation += "Facing Right"
    else:
        yaw_orientation += "Facing Left"

    if abs(pitch) > 15:
        if pitch < 0:
            pitch_orientation += "Looking Down"
        else:
            pitch_orientation += "Looking Up"
    if abs(roll) > 15:
        if roll < 0:
            roll_orientation += "Tilted Left"
        else:
            roll_orientation += "Tilted Right"

    return yaw_orientation,pitch_orientation,roll_orientation



def process( img_path, output_path):
    # read image
    frame =cv2.imread(img_path)
    original_height, original_width, _ = frame.shape




    # face detection
    dets = detector(frame, 0)
    if 0 == len(dets):
        print("ERROR: found no face")
        return
    largest_index = _largest_face(dets)
    face_rectangle = dets[largest_index]

    x1,y1,x2,y2= face_rectangle.left(),face_rectangle.top(),face_rectangle.right(),face_rectangle.bottom()
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

    # get all  the landmarks points within specific face
    landmarks= predictor(frame, face_rectangle)


    # calculate head pose
    pitch, yaw, roll, p1, p2 = get_head_pose(landmarks, frame)

    if pitch is not None and yaw is not None and roll is not None:
        # visualize : nose tip to projected points
        cv2.line(frame, p1, p2, (255, 0, 0), 2)
        #cv2.putText(frame, f"Pitch: {pitch:.2f}", (x, y - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #cv2.putText(frame, f"Yaw: {yaw:.2f}", (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #cv2.putText(frame, f"Roll: {roll:.2f}", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        Facing_state, updown_state, tilted_state = determine_head_pose_state(pitch, yaw, roll)
        cv2.putText(frame, Facing_state, (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, updown_state, (x1-160, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #cv2.putText(frame, tilted_state, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    cv2.imwrite(output_path,frame)






if __name__ == '__main__':

    img_path = 'test2.png'
    output_path = 'result2_2.png'
    process(img_path, output_path)