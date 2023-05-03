import cv2
import numpy as np
import pytesseract
import xlsxwriter

import helper

mode_dict = {"single_word": 8,
             "single character": 10,
             "all": 6
             }


def read_image(path):
    """
    Read image from path, covert it to grayscale and return it as numpy array
    :param path: path to the image
    :return: np array in shape of (n, m, 1)
    """
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def convert_to_binary(img):
    """
    convert the image to binary image
    :param img: np array in shape of (n, m, 1)
    :return: np array in shape of (n, m, 1)
    """

    # apply thresholding

    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 21, 5)
    # perfrom closing to remove the noise
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    return img


def text_extraction(img, mode):
    """
    extract the text from the image
    :param img: np array in shape of (n, m, 1)
    :return: string if text was found, int if num, else None
    """

    # set the path to the tesseract executable
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    # # convert the image to binary
    # img = convert_to_binary(img)

    # resize the image
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # apply erosion
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)

    data = None

    text = pytesseract.image_to_string(img, lang='eng',
                                       config=f'--psm {mode_dict[mode]}')
    num = pytesseract.image_to_string(img, lang='eng',
                                      config='--psm 8 -c tessedit_char_whitelist=0123456789')
    # prefer numbers
    if num:
        data = int(num[:-1])  # remove the \n at the end of the string

    elif text:
        data = text[:-1]  # remove the \n at the end of the string

    return data


def insert_to_excel(data, num_row,  out_path):
    """
    insert the output to Excel file
    :param data:list of string containing the data, each string is a cell
    :param num_row: number of rows in the Excel file
    :param out_path: path to the output file
    :return: None
    """

    # create file in the output file
    workbook = xlsxwriter.Workbook(out_path)
    worksheet = workbook.add_worksheet()

    # insert the data to the file
    row = 0
    col = 0
    for cell in data:
        worksheet.write(row, col, cell)
        row += 1
        if row == num_row:
            row = 0
            col += 1

    # close the file
    workbook.close()


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    bounding_boxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, bounding_boxes) = zip(*sorted(zip(cnts, bounding_boxes),
                                         key=lambda b: b[1][i],
                                         reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, bounding_boxes)


# Functon for extracting the box
def detect_grid(img_bin, is_handwritten=False):
    """
    detect the grid in the image
    :param img_bin: np array in shape of (n, m, 1)
    :param is_handwritten: bool, True if the image is handwritten
    :return: image with the grid
    """

    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    if is_handwritten:
        # apply morphology operations to connect the lines
        img_bin = cv2.erode(img_bin, kernel, iterations=3)

        cv2.imwrite("Images/Image_erode.jpg", img_bin)

    img_bin = 255 - img_bin  # Invert the image

    # Defining a kernel length
    kernel_length = np.array(img_bin).shape[1] // 20

    # A vertical kernel of (1 X kernel_length)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                                (1, kernel_length))

    # A horizontal kernel of (kernel_length X 1)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                                  (kernel_length, 1))

    # Morphological operation to detect vertical lines from an image
    vertical_lines_img = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN,
                                          vertical_kernel,
                                          iterations=3)

    cv2.imwrite("Images/vertical_lines.jpg", vertical_lines_img)

    # Morphological operation to detect horizontal lines from an image
    horizontal_lines_img = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN,
                                            horizontal_kernel,
                                            iterations=3)

    cv2.imwrite("Images/horizontal_lines.jpg", horizontal_lines_img)

    # add both the images and use it to find the contours
    img_final_bin = vertical_lines_img + horizontal_lines_img

    # perform dilation to connect the lines
    img_final_bin = cv2.dilate(img_final_bin, kernel, iterations=3)

    return img_final_bin


def find_cell_and_extarct_data(cropped_dir_path, img_grid, img):
    # Find contours for image, which will detect all the boxes
    contours, hierarchy = cv2.findContours(
        img_grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Sort all the contours by top to bottom.
    (contours, boundingBoxes) = sort_contours(contours, method="right-to-left")

    idx = 0
    cells = []
    for c in boundingBoxes:
        # Returns the location and width,height for every contour
        x, y, w, h = c

        # If the cell height is greater than 10, width is >10,
        # and proportion is between 0.5 and 2
        if 2 > (w // h) > 0.5 and w > 10 and h > 10:
            idx += 1
            crop_img = img[y:y + h, x:x + w]

            data = text_extraction(crop_img, "single_word")
            cells.append(data)
            cv2.imwrite(f"{cropped_dir_path}{idx}_{data}.png", crop_img)

    # For Debugging
    # Enable this line to see all contours.
    for i in range(len(contours)):
        tmp = img.copy()
        cv2.drawContours(tmp, contours, i, (0, 0, 255), 3)
        cv2.imwrite(f"Images/contour{i}.jpg", tmp)

    cv2.imwrite("img_contour.jpg", img)

    return cells


if __name__ == '__main__':
    path = "test_image/img5.jpeg"
    is_handwritten = True

    helper.delete_dir_content("images/")
    helper.delete_dir_content("cells/")

    # read the image
    img = read_image(path)

    # convert to binary
    img_bin = convert_to_binary(img)
    cv2.imwrite("Images/img_bin.jpg", img_bin)

    # extract the box
    img_grid = detect_grid(img_bin, is_handwritten=is_handwritten)
    cv2.imwrite("Images/img_grid.jpg", img_grid)

    # extract the grid from the image
    r = np.logical_not(img_bin).astype(np.uint8) * 255 - img_grid
    # invert r
    r = 255 - r

    # for debugging
    cv2.imwrite("Images/img_minus_grid.jpg", r)

    # find the cells and extract the data
    cells = find_cell_and_extarct_data("cells/", img_grid, r)

    # write the data to excel
    insert_to_excel(data=cells, num_row=10, out_path=f"{path}.xlsx")
