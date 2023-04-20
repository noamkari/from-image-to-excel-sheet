import cv2
import numpy as np
import os
import shutil


def write_cells(img, w, h):
    """
    write the cells on the image
    :param img:
    :param w: width of cell
    :param h: height of cell
    :return:
    """
    for i in range(0, 100):
        cv2.line(img, (0, i * h), (img.shape[1], i * h), (255, 0, 0), 1)
        cv2.line(img, (i * w, 0), (i * w, img.shape[0]), (255, 0, 0), 1)

    cv2.imwrite("img_with_cells.jpg", img)


def color_line(lines, img):
    # convert image to color
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # color the lines in different color on the original image
    colors_r = np.linspace(0, 255, len(lines))

    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # show the image with the lines
    show_image(img)


def color_intercection(intercection, img):
    # convert image to color
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # color the lines in different color on the original image
    colors_r = np.linspace(0, 255, len(intercection))

    for i, (x, y) in enumerate(intercection):
        cv2.circle(img, (x, y), 5, (0, 255, 0), 2)

    # show the image with the lines
    show_image(img)


def show_image(img, name='Image'):
    cv2.imshow(name, img)
    cv2.moveWindow(name, 500, 0)
    cv2.waitKey(0)
    cv2.destroyWindow(name)


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


# Return true if line segments AB and CD intersect
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def angle(line1, line2):
    """
    calculate the angle between the two lines
    :param line1:
    :param line2:
    :return: int angle in degrees
    """
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]

    # calculate the angle between the lines
    a = np.array([x2 - x1, y2 - y1])
    b = np.array([x4 - x3, y4 - y3])
    angle = np.math.atan2(np.linalg.det([a, b]), np.dot(a, b))

    return np.degrees(angle)


def find_intersection_between_lines(lines):
    """
    find the intersection between the lines
    :param lines: ndarray of shape (n, 1, 4)
    :return:
    """
    intersections = []

    for i in range(len(lines)):
        x1, y1, x2, y2 = lines[i][0]
        for j in range(i + 1, len(lines)):

            x3, y3, x4, y4 = lines[j][0]

            # check if the lines are crossing
            if not intersect([x1, y1], [x2, y2], [x3, y3], [x4, y4]):
                continue

            # calculate the intersection point
            x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (
                    x3 * y4 - y3 * x4)) / (
                        (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
            y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (
                    x3 * y4 - y3 * x4)) / (
                        (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))

            intersections.append((int(x), int(y)))

    return intersections


def delete_dir_content(name):
    """
    delete all the files in the directory
    :param name:
    :return: None
    """
    for filename in os.listdir(name):
        file_path = os.path.join(name, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def rotate_image(img, degree):
    """
    rotate the image by given degree
    :param img: numpy array of image
    :param degree:
    :return:
    """
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), degree, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst
