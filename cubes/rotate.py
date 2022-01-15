from scipy import stats
import numpy as np
import sys
import cv2
import math

def detect_face(bgr_image_input):

    gray = cv2.cvtColor(bgr_image_input,cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    gray = cv2.adaptiveThreshold(gray,20,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,5,0)
    #cv2.imwrite()
    try:
         _, contours, hierarchy = cv2.findContours(gray,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    except:
         contours, hierarchy = cv2.findContours(gray,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)


    i = 0
    contour_id = 0
    #print(len(contours))
    count = 0
    blob_colors = []
    for contour in contours:
        A1 = cv2.contourArea(contour)
        contour_id = contour_id + 1

        if A1 < 3000 and A1 > 1000:
            perimeter = cv2.arcLength(contour, True)
            epsilon = 0.01 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            hull = cv2.convexHull(contour)
            if cv2.norm(((perimeter / 4) * (perimeter / 4)) - A1) < 150:
                #if cv2.ma
                count = count + 1
                x, y, w, h = cv2.boundingRect(contour)
                #cv2.rectangle(bgr_image_input, (x, y), (x + w, y + h), (0, 255, 255), 2)
                #cv2.imshow('cutted contour', bgr_image_input[y:y + h, x:x + w])
                val = (50*y) + (10*x)
                blob_color = np.array(cv2.mean(bgr_image_input[y:y+h,x:x+w])).astype(int)
                cv2.drawContours(bgr_image_input,[contour],0,(255, 255, 0),2)
                cv2.drawContours(bgr_image_input, [approx], 0, (255, 255, 0), 2)
                blob_color = np.append(blob_color, val)
                blob_color = np.append(blob_color, x)
                blob_color = np.append(blob_color, y)
                blob_color = np.append(blob_color, w)
                blob_color = np.append(blob_color, h)
                blob_colors.append(blob_color)
    if len(blob_colors) > 0:
        blob_colors = np.asarray(blob_colors)
        blob_colors = blob_colors[blob_colors[:, 4].argsort()]
    face = np.array([0,0,0,0,0,0,0,0,0])
    if len(blob_colors) == 9:
        #print(blob_colors)
        for i in range(9):
            #print(blob_colors[i])
            if blob_colors[i][0] > 120 and blob_colors[i][1] > 120 and blob_colors[i][2] > 100:
                blob_colors[i][3] = 1
                face[i] = 1
            elif blob_colors[i][0] < 100 and blob_colors[i][1] > 120 and blob_colors[i][2] > 120 and np.abs(blob_colors[i][1]-blob_colors[i][2])<30:
                blob_colors[i][3] = 2
                face[i] = 2
            elif blob_colors[i][0] > blob_colors[i][1] and blob_colors[i][1] > blob_colors[i][2]:
                blob_colors[i][3] = 3
                face[i] = 3
            elif blob_colors[i][1] > blob_colors[i][0] and blob_colors[i][1] > blob_colors[i][2] and np.abs(blob_colors[i][0] - blob_colors[i][2]) < 30:
                blob_colors[i][3] = 4
                face[i] = 4
            elif blob_colors[i][2] > blob_colors[i][0] and blob_colors[i][2] > blob_colors[i][1] and np.abs(blob_colors[i][0] - blob_colors[i][1]) < 30 and blob_colors[i][0] < 80:
                blob_colors[i][3] = 5
                face[i] = 5
            elif blob_colors[i][1] < blob_colors[i][2] and blob_colors[i][0] < blob_colors[i][1] and blob_colors[i][2] > 120:
                blob_colors[i][3] = 6
                face[i] = 6
        #print(face)
        if np.count_nonzero(face) == 9:
            #print(face)
            #print (blob_colors)
            return face, blob_colors
        else:
            return [0,0], blob_colors
    else:
        return [0,0,0], blob_colors
        #break

def rotate_cw(face):
    final = np.copy(face)
    final[0, 0] = face[0, 6]
    final[0, 1] = face[0, 3]
    final[0, 2] = face[0, 0]
    final[0, 3] = face[0, 7]
    final[0, 4] = face[0, 4]
    final[0, 5] = face[0, 1]
    final[0, 6] = face[0, 8]
    final[0, 7] = face[0, 5]
    final[0, 8] = face[0, 2]
    return final

def rotate_ccw(face):
    final = np.copy(face)
    final[0, 8] = face[0, 6]
    final[0, 7] = face[0, 3]
    final[0, 6] = face[0, 0]
    final[0, 5] = face[0, 7]
    final[0, 4] = face[0, 4]
    final[0, 3] = face[0, 1]
    final[0, 2] = face[0, 8]
    final[0, 1] = face[0, 5]
    final[0, 0] = face[0, 2]
    return final

def right_cw(video,videoWriter,up_face,right_face,front_face,down_face,left_face,back_face):
    print("Next Move: R Clockwise")
    temp = np.copy(front_face)
    front_face[0, 2] = down_face[0, 2]
    front_face[0, 5] = down_face[0, 5]
    front_face[0, 8] = down_face[0, 8]
    down_face[0, 2] = back_face[0, 6]
    down_face[0, 5] = back_face[0, 3]
    down_face[0, 8] = back_face[0, 0]
    back_face[0, 0] = up_face[0, 8]
    back_face[0, 3] = up_face[0, 5]
    back_face[0, 6] = up_face[0, 2]
    up_face[0, 2] = temp[0, 2]
    up_face[0, 5] = temp[0, 5]
    up_face[0, 8] = temp[0, 8]
    right_face = rotate_cw(right_face)
    #front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, bgr_image_input = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, blob_colors = detect_face(bgr_image_input)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return up_face,right_face,front_face,down_face,left_face,back_face
                elif np.array_equal(detected_face,temp) == True:
                    centroid1 = blob_colors[8]
                    centroid2 = blob_colors[2]
                    point1 = (centroid1[5]+(centroid1[7]/2), centroid1[6]+(centroid1[7]/2))
                    point2 = (centroid2[5]+(centroid2[8]/2), centroid2[6]+(centroid2[8]/2))
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 255), 4, tipLength=0.2)
        videoWriter.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

def right_ccw(video, videoWriter, up_face,right_face,front_face,down_face,left_face,back_face):
    print("Next Move: R CounterClockwise")
    temp = np.copy(front_face)
    front_face[0, 2] = up_face[0, 2]
    front_face[0, 5] = up_face[0, 5]
    front_face[0, 8] = up_face[0, 8]
    up_face[0, 2] = back_face[0, 6]
    up_face[0, 5] = back_face[0, 3]
    up_face[0, 8] = back_face[0, 0]
    back_face[0, 0] = down_face[0, 8]
    back_face[0, 3] = down_face[0, 5]
    back_face[0, 6] = down_face[0, 2]
    down_face[0, 2] = temp[0, 2]
    down_face[0, 5] = temp[0, 5]
    down_face[0, 8] = temp[0, 8]
    right_face = rotate_ccw(right_face)
    # front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, bgr_image_input = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, blob_colors = detect_face(bgr_image_input)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return up_face,right_face,front_face,down_face,left_face,back_face
                elif np.array_equal(detected_face,temp) == True:
                    centroid1 = blob_colors[2]
                    centroid2 = blob_colors[8]
                    point1 = (centroid1[5]+(centroid1[7]/2), centroid1[6]+(centroid1[7]/2))
                    point2 = (centroid2[5]+(centroid2[8]/2), centroid2[6]+(centroid2[8]/2))
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 255), 4, tipLength=0.2)
        videoWriter.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

def left_cw(video,videoWriter,up_face,right_face,front_face,down_face,left_face,back_face):
    print("Next Move: L Clockwise")
    temp = np.copy(front_face)
    front_face[0, 0] = up_face[0, 0]
    front_face[0, 3] = up_face[0, 3]
    front_face[0, 6] = up_face[0, 6]
    up_face[0, 0] = back_face[0, 8]
    up_face[0, 3] = back_face[0, 5]
    up_face[0, 6] = back_face[0, 2]
    back_face[0, 2] = down_face[0, 6]
    back_face[0, 5] = down_face[0, 3]
    back_face[0, 8] = down_face[0, 0]
    down_face[0, 0] = temp[0, 0]
    down_face[0, 3] = temp[0, 3]
    down_face[0, 6] = temp[0, 6]
    left_face = rotate_cw(left_face)
    #front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, bgr_image_input = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, blob_colors = detect_face(bgr_image_input)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return up_face,right_face,front_face,down_face,left_face,back_face
                elif np.array_equal(detected_face,temp) == True:
                    centroid1 = blob_colors[0]
                    centroid2 = blob_colors[6]
                    point1 = (centroid1[5]+(centroid1[7]/2), centroid1[6]+(centroid1[7]/2))
                    point2 = (centroid2[5]+(centroid2[8]/2), centroid2[6]+(centroid2[8]/2))
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 255), 4, tipLength=0.2)
        videoWriter.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

def left_ccw(video,videoWriter,up_face,right_face,front_face,down_face,left_face,back_face):
    print("Next Move: L CounterClockwise")
    temp = np.copy(front_face)
    front_face[0, 0] = down_face[0, 0]
    front_face[0, 3] = down_face[0, 3]
    front_face[0, 6] = down_face[0, 6]
    down_face[0, 0] = back_face[0, 8]
    down_face[0, 3] = back_face[0, 5]
    down_face[0, 6] = back_face[0, 2]
    back_face[0, 2] = up_face[0, 6]
    back_face[0, 5] = up_face[0, 3]
    back_face[0, 8] = up_face[0, 0]
    up_face[0, 0] = temp[0, 0]
    up_face[0, 3] = temp[0, 3]
    up_face[0, 6] = temp[0, 6]
    left_face = rotate_ccw(left_face)
    #front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, bgr_image_input = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, blob_colors = detect_face(bgr_image_input)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return up_face,right_face,front_face,down_face,left_face,back_face
                elif np.array_equal(detected_face,temp) == True:
                    centroid1 = blob_colors[6]
                    centroid2 = blob_colors[0]
                    point1 = (centroid1[5]+(centroid1[7]/2), centroid1[6]+(centroid1[7]/2))
                    point2 = (centroid2[5]+(centroid2[8]/2), centroid2[6]+(centroid2[8]/2))
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 255), 4, tipLength=0.2)
        videoWriter.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

def front_cw(video,videoWriter,up_face,right_face,front_face,down_face,left_face,back_face):
    print(front_face)
    print("Next Move: F Clockwise")
    temp1 = np.copy(front_face)
    temp = np.copy(up_face)
    front_face = rotate_cw(front_face)
    temp2 = np.copy(front_face)
    if np.array_equal(temp2, temp1) == True:
        [up_face, right_face, front_face, down_face, left_face, back_face] = turn_to_right(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face)
        [up_face, right_face, front_face, down_face, left_face, back_face] = left_cw(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face)
        [up_face, right_face, front_face, down_face, left_face, back_face] = turn_to_front(video, videoWriter, up_face, right_face, front_face, down_face, left_face, back_face)
        return up_face, right_face, front_face, down_face, left_face, back_face
    up_face[0, 8] = left_face[0, 2]
    up_face[0, 7] = left_face[0, 5]
    up_face[0, 6] = left_face[0, 8]
    left_face[0, 2] = down_face[0, 0]
    left_face[0, 5] = down_face[0, 1]
    left_face[0, 8] = down_face[0, 2]
    down_face[0, 2] = right_face[0, 0]
    down_face[0, 1] = right_face[0, 3]
    down_face[0, 0] = right_face[0, 6]
    right_face[0, 0] = temp[0, 6]
    right_face[0, 3] = temp[0, 7]
    right_face[0, 6] = temp[0, 8]

    #front_face = temp

    print(front_face)
    faces = []
    while True:

        is_ok, bgr_image_input = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, blob_colors = detect_face(bgr_image_input)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return up_face,right_face,front_face,down_face,left_face,back_face
                elif np.array_equal(detected_face,temp1) == True:
                    centroid1 = blob_colors[8]
                    centroid2 = blob_colors[6]
                    centroid3 = blob_colors[0]
                    centroid4 = blob_colors[2]
                    point1 = (centroid1[5] + (centroid1[7] / 4), centroid1[6] + (centroid1[7] / 2))
                    point2 = (centroid2[5] + (3 * centroid2[8] / 4), centroid2[6] + (centroid2[8] / 2))
                    point3 = (centroid2[5] + (centroid2[7] / 2), centroid2[6] + (centroid2[7] / 4))
                    point4 = (centroid3[5] + (centroid3[8] / 2), centroid3[6] + (3 * centroid3[8] / 4))
                    point5 = (centroid3[5] + (3 * centroid3[8] / 4), centroid3[6] + (centroid3[8] / 2))
                    point6 = (centroid4[5] + (centroid4[8] / 4), centroid4[6] + (centroid4[8] / 2))
                    point7 = (centroid4[5] + (centroid4[8] / 2), centroid4[6] + (3 * centroid4[8] / 4))
                    point8 = (centroid1[5] + (centroid1[8] / 2), centroid1[6] + (centroid1[8] / 4))
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point3, point4, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point5, point6, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point7, point8, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 255), 4, tipLength=0.2)
                    cv2.arrowedLine(bgr_image_input, point3, point4, (0, 0, 255), 4, tipLength=0.2)
                    cv2.arrowedLine(bgr_image_input, point5, point6, (0, 0, 255), 4, tipLength=0.2)
                    cv2.arrowedLine(bgr_image_input, point7, point8, (0, 0, 255), 4, tipLength=0.2)
        videoWriter.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

def front_ccw(video,videoWriter,up_face,right_face,front_face,down_face,left_face,back_face):
    print("Next Move: F CounterClockwise")
    temp = np.copy(up_face)
    temp1 = np.copy(front_face)
    front_face = rotate_ccw(front_face)
    temp2 = np.copy(front_face)
    if np.array_equal(temp2,temp1) == True:
            [up_face, right_face, front_face, down_face, left_face, back_face] = turn_to_right(video,videoWriter,up_face,right_face,front_face,down_face,left_face,back_face)
            [up_face, right_face, front_face, down_face, left_face, back_face] = left_ccw(video,videoWriter,up_face,right_face,front_face,down_face,left_face,back_face)
            [up_face, right_face, front_face, down_face, left_face, back_face] = turn_to_front(video,videoWriter,up_face,right_face,front_face,down_face,left_face,back_face)
            return up_face,right_face,front_face,down_face,left_face,back_face
    up_face[0, 6] = right_face[0, 0]
    up_face[0, 7] = right_face[0, 3]
    up_face[0, 8] = right_face[0, 6]
    right_face[0, 0] = down_face[0, 2]
    right_face[0, 3] = down_face[0, 1]
    right_face[0, 6] = down_face[0, 0]
    down_face[0, 0] = left_face[0, 2]
    down_face[0, 1] = left_face[0, 5]
    down_face[0, 2] = left_face[0, 8]
    left_face[0, 8] = temp[0, 6]
    left_face[0, 5] = temp[0, 7]
    left_face[0, 2] = temp[0, 8]

    #front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, bgr_image_input = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, blob_colors = detect_face(bgr_image_input)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return up_face,right_face,front_face,down_face,left_face,back_face
                elif np.array_equal(detected_face,temp1) == True:
                    centroid1 = blob_colors[2]
                    centroid2 = blob_colors[0]
                    centroid3 = blob_colors[6]
                    centroid4 = blob_colors[8]
                    point1 = (centroid1[5] + (centroid1[7] / 4), centroid1[6] + (centroid1[7] / 2))
                    point2 = (centroid2[5] + (3 * centroid2[8]/4), centroid2[6] + (centroid2[8] / 2))
                    point3 = (centroid2[5] + (centroid2[7] / 2), centroid2[6] + (3 * centroid2[7] / 4))
                    point4 = (centroid3[5] + (centroid3[8] / 2), centroid3[6] + (centroid3[8] / 4))
                    point5 = (centroid3[5] + (3 * centroid3[8] / 4), centroid3[6] + (centroid3[8] / 2))
                    point6 = (centroid4[5] + (centroid4[8] / 4), centroid4[6] + (centroid4[8] / 2))
                    point7 = (centroid4[5] + (centroid4[8] / 2), centroid4[6] + (centroid4[8] / 4))
                    point8 = (centroid1[5] + (centroid1[8] / 2), centroid1[6] + (3 * centroid1[8] / 4))
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point3, point4, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point5, point6, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point7, point8, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 255), 4, tipLength=0.2)
                    cv2.arrowedLine(bgr_image_input, point3, point4, (0, 0, 255), 4, tipLength=0.2)
                    cv2.arrowedLine(bgr_image_input, point5, point6, (0, 0, 255), 4, tipLength=0.2)
                    cv2.arrowedLine(bgr_image_input, point7, point8, (0, 0, 255), 4, tipLength=0.2)
        videoWriter.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

def back_cw(video,videoWriter,up_face,right_face,front_face,down_face,left_face,back_face):
    print("Next Move: B Clockwise")
    temp = np.copy(up_face)
    up_face[0, 0] = right_face[0, 2]
    up_face[0, 1] = right_face[0, 5]
    up_face[0, 2] = right_face[0, 8]
    right_face[0, 8] = down_face[0, 6]
    right_face[0, 5] = down_face[0, 7]
    right_face[0, 2] = down_face[0, 8]
    down_face[0, 6] = left_face[0, 0]
    down_face[0, 7] = left_face[0, 3]
    down_face[0, 8] = left_face[0, 6]
    left_face[0, 0] = temp[0, 2]
    left_face[0, 3] = temp[0, 1]
    left_face[0, 6] = temp[0, 0]
    back_face = rotate_cw(back_face)
    #front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, bgr_image_input = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, blob_colors = detect_face(bgr_image_input)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return up_face,right_face,front_face,down_face,left_face,back_face
        videoWriter.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

def back_ccw(video,videoWriter,up_face,right_face,front_face,down_face,left_face,back_face):
    print("Next Move: B CounterClockwise")
    temp = np.copy(up_face)
    up_face[0, 2] = left_face[0, 0]
    up_face[0, 1] = left_face[0, 3]
    up_face[0, 0] = left_face[0, 6]
    left_face[0, 0] = down_face[0, 6]
    left_face[0, 3] = down_face[0, 7]
    left_face[0, 6] = down_face[0, 8]
    down_face[0, 6] = right_face[0, 8]
    down_face[0, 7] = right_face[0, 5]
    down_face[0, 8] = right_face[0, 2]
    right_face[0, 2] = temp[0, 0]
    right_face[0, 5] = temp[0, 1]
    right_face[0, 8] = temp[0, 2]
    back_face = rotate_ccw(back_face)
    #front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, bgr_image_input = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, blob_colors = detect_face(bgr_image_input)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return up_face,right_face,front_face,down_face,left_face,back_face
        videoWriter.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

def up_cw(video,videoWriter,up_face,right_face,front_face,down_face,left_face,back_face):
    print("Next Move: U Clockwise")
    temp = np.copy(front_face)
    front_face[0, 0] = right_face[0, 0]
    front_face[0, 1] = right_face[0, 1]
    front_face[0, 2] = right_face[0, 2]
    right_face[0, 0] = back_face[0, 0]
    right_face[0, 1] = back_face[0, 1]
    right_face[0, 2] = back_face[0, 2]
    back_face[0, 0] = left_face[0, 0]
    back_face[0, 1] = left_face[0, 1]
    back_face[0, 2] = left_face[0, 2]
    left_face[0, 0] = temp[0, 0]
    left_face[0, 1] = temp[0, 1]
    left_face[0, 2] = temp[0, 2]
    up_face = rotate_cw(up_face)
    #front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, bgr_image_input = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, blob_colors = detect_face(bgr_image_input)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return up_face,right_face,front_face,down_face,left_face,back_face
                elif np.array_equal(detected_face,temp) == True:
                    centroid1 = blob_colors[2]
                    centroid2 = blob_colors[0]
                    point1 = (centroid1[5]+(centroid1[7]/2), centroid1[6]+(centroid1[7]/2))
                    point2 = (centroid2[5]+(centroid2[8]/2), centroid2[6]+(centroid2[8]/2))
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 255), 4, tipLength=0.2)
        videoWriter.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

def up_ccw(video,videoWriter,up_face,right_face,front_face,down_face,left_face,back_face):
    print("Next Move: U CounterClockwise")
    temp = np.copy(front_face)
    front_face[0, 0] = left_face[0, 0]
    front_face[0, 1] = left_face[0, 1]
    front_face[0, 2] = left_face[0, 2]
    left_face[0, 0] = back_face[0, 0]
    left_face[0, 1] = back_face[0, 1]
    left_face[0, 2] = back_face[0, 2]
    back_face[0, 0] = right_face[0, 0]
    back_face[0, 1] = right_face[0, 1]
    back_face[0, 2] = right_face[0, 2]
    right_face[0, 0] = temp[0, 0]
    right_face[0, 1] = temp[0, 1]
    right_face[0, 2] = temp[0, 2]
    up_face = rotate_ccw(up_face)
    #front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, bgr_image_input = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, blob_colors = detect_face(bgr_image_input)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return up_face,right_face,front_face,down_face,left_face,back_face
                elif np.array_equal(detected_face,temp) == True:
                    centroid1 = blob_colors[0]
                    centroid2 = blob_colors[2]
                    point1 = (centroid1[5]+(centroid1[7]/2), centroid1[6]+(centroid1[7]/2))
                    point2 = (centroid2[5]+(centroid2[8]/2), centroid2[6]+(centroid2[8]/2))
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 255), 4, tipLength=0.2)
        videoWriter.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

def down_cw(video,videoWriter,up_face,right_face,front_face,down_face,left_face,back_face):
    print("Next Move: D Clockwise")
    temp = np.copy(front_face)
    front_face[0, 6] = left_face[0, 6]
    front_face[0, 7] = left_face[0, 7]
    front_face[0, 8] = left_face[0, 8]
    left_face[0, 6] = back_face[0, 6]
    left_face[0, 7] = back_face[0, 7]
    left_face[0, 8] = back_face[0, 8]
    back_face[0, 6] = right_face[0, 6]
    back_face[0, 7] = right_face[0, 7]
    back_face[0, 8] = right_face[0, 8]
    right_face[0, 6] = temp[0, 6]
    right_face[0, 7] = temp[0, 7]
    right_face[0, 8] = temp[0, 8]
    down_face = rotate_cw(down_face)
    #front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, bgr_image_input = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, blob_colors = detect_face(bgr_image_input)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return up_face,right_face,front_face,down_face,left_face,back_face
                elif np.array_equal(detected_face,temp) == True:
                    centroid1 = blob_colors[6]
                    centroid2 = blob_colors[8]
                    point1 = (centroid1[5]+(centroid1[7]/2), centroid1[6]+(centroid1[7]/2))
                    point2 = (centroid2[5]+(centroid2[8]/2), centroid2[6]+(centroid2[8]/2))
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 255), 4, tipLength=0.2)
        videoWriter.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

def down_ccw(video,videoWriter,up_face,right_face,front_face,down_face,left_face,back_face):
    print("Next Move: D CounterClockwise")
    temp = np.copy(front_face)
    front_face[0, 6] = right_face[0, 6]
    front_face[0, 7] = right_face[0, 7]
    front_face[0, 8] = right_face[0, 8]
    right_face[0, 6] = back_face[0, 6]
    right_face[0, 7] = back_face[0, 7]
    right_face[0, 8] = back_face[0, 8]
    back_face[0, 6] = left_face[0, 6]
    back_face[0, 7] = left_face[0, 7]
    back_face[0, 8] = left_face[0, 8]
    left_face[0, 6] = temp[0, 6]
    left_face[0, 7] = temp[0, 7]
    left_face[0, 8] = temp[0, 8]
    down_face = rotate_ccw(down_face)
    #front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, bgr_image_input = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, blob_colors = detect_face(bgr_image_input)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return up_face,right_face,front_face,down_face,left_face,back_face
                elif np.array_equal(detected_face,temp) == True:
                    centroid1 = blob_colors[8]
                    centroid2 = blob_colors[6]
                    point1 = (centroid1[5]+(centroid1[7]/2), centroid1[6]+(centroid1[7]/2))
                    point2 = (centroid2[5]+(centroid2[8]/2), centroid2[6]+(centroid2[8]/2))
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 255), 4, tipLength=0.2)
        videoWriter.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

def turn_to_right(video,videoWriter,up_face,right_face,front_face,down_face,left_face,back_face):
    print("Next Move: Show Right Face")
    temp = np.copy(front_face)
    front_face = np.copy(right_face)
    right_face = np.copy(back_face)
    back_face = np.copy(left_face)
    left_face = np.copy(temp)
    up_face = rotate_cw(up_face)
    down_face = rotate_ccw(down_face)
    #front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, bgr_image_input = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, blob_colors = detect_face(bgr_image_input)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return up_face,right_face,front_face,down_face,left_face,back_face
                elif np.array_equal(detected_face,temp) == True:
                    centroid1 = blob_colors[8]
                    centroid2 = blob_colors[6]
                    centroid3 = blob_colors[5]
                    centroid4 = blob_colors[3]
                    centroid5 = blob_colors[2]
                    centroid6 = blob_colors[0]
                    point1 = (centroid1[5] + (centroid1[7] / 2), centroid1[6] + (centroid1[7] / 2))
                    point2 = (centroid2[5] + (centroid2[8] / 2), centroid2[6] + (centroid2[8] / 2))
                    point3 = (centroid3[5] + (centroid3[7] / 2), centroid3[6] + (centroid3[7] / 2))
                    point4 = (centroid4[5] + (centroid4[8] / 2), centroid4[6] + (centroid4[8] / 2))
                    point5 = (centroid5[5] + (centroid5[7] / 2), centroid5[6] + (centroid5[7] / 2))
                    point6 = (centroid6[5] + (centroid6[8] / 2), centroid6[6] + (centroid6[8] / 2))
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point3, point4, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point5, point6, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 255), 4, tipLength=0.2)
                    cv2.arrowedLine(bgr_image_input, point3, point4, (0, 0, 255), 4, tipLength=0.2)
                    cv2.arrowedLine(bgr_image_input, point5, point6, (0, 0, 255), 4, tipLength=0.2)
        videoWriter.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break

def turn_to_front(video,videoWriter,up_face,right_face,front_face,down_face,left_face,back_face):
    print("Next Move: Show Front Face")
    temp = np.copy(front_face)
    front_face = np.copy(left_face)
    left_face = np.copy(back_face)
    back_face = np.copy(right_face)
    right_face = np.copy(temp)
    up_face = rotate_ccw(up_face)
    down_face = rotate_cw(down_face)
    #front_face = temp

    print(front_face)
    faces = []
    while True:
        is_ok, bgr_image_input = video.read()

        if not is_ok:
            print("Cannot read video source")
            sys.exit()

        face, blob_colors = detect_face(bgr_image_input)
        # print(len(face))
        if len(face) == 9:
            faces.append(face)
            if len(faces) == 10:
                face_array = np.array(faces)
                # print('INNNNN')
                # face_array = np.transpose(face_array)
                detected_face = stats.mode(face_array)[0]
                up_face = np.asarray(up_face)
                front_face = np.asarray(front_face)
                detected_face = np.asarray(detected_face)
                faces = []
                if np.array_equal(detected_face, front_face) == True:
                    print("MOVE MADE")
                    return up_face,right_face,front_face,down_face,left_face,back_face
                elif np.array_equal(detected_face,temp) == True:
                    centroid1 = blob_colors[6]
                    centroid2 = blob_colors[8]
                    centroid3 = blob_colors[3]
                    centroid4 = blob_colors[5]
                    centroid5 = blob_colors[0]
                    centroid6 = blob_colors[2]
                    point1 = (centroid1[5] + (centroid1[7] / 2), centroid1[6] + (centroid1[7] / 2))
                    point2 = (centroid2[5] + (centroid2[8] / 2), centroid2[6] + (centroid2[8] / 2))
                    point3 = (centroid3[5] + (centroid3[7] / 2), centroid3[6] + (centroid3[7] / 2))
                    point4 = (centroid4[5] + (centroid4[8] / 2), centroid4[6] + (centroid4[8] / 2))
                    point5 = (centroid5[5] + (centroid5[7] / 2), centroid5[6] + (centroid5[7] / 2))
                    point6 = (centroid6[5] + (centroid6[8] / 2), centroid6[6] + (centroid6[8] / 2))
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point3, point4, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point5, point6, (0, 0, 0), 7, tipLength = 0.2)
                    cv2.arrowedLine(bgr_image_input, point1, point2, (0, 0, 255), 4, tipLength=0.2)
                    cv2.arrowedLine(bgr_image_input, point3, point4, (0, 0, 255), 4, tipLength=0.2)
                    cv2.arrowedLine(bgr_image_input, point5, point6, (0, 0, 255), 4, tipLength=0.2)
        videoWriter.write(bgr_image_input)
        cv2.imshow("Output Image", bgr_image_input)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27 or key_pressed == ord('q'):
            break
