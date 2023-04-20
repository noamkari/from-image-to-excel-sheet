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

    # convert the image to binary
    img = convert_to_binary(img)

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


def insert_to_excel(data, out_path):
    """
    insert the output to Excel file
    :param data:list of string containing the data, each string is a cell
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
        if row == 20:
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
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i],
                                        reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


# Functon for extracting the box
def detect_grid(img_bin, is_handwritten=False):
    """
    extract the boxes from the image
    :param is_handwritten:
    :param img_for_box_extraction_path:
    :param cropped_dir_path:
    :return:
    """

    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    if is_handwritten:
        # apply morphology operations to connect the lines
        img_bin = cv2.erode(img_bin, kernel, iterations=3)

        cv2.imwrite("Images/Image_erode.jpg", img_bin)

    img_bin = 255 - img_bin  # Invert the image

    # Defining a kernel length
    kernel_length = np.array(img_bin).shape[1] // 40

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


def extract_number_of_row_and_col(bin_img):
    """
    extract the number of rows and columns from the image
    :param bin_img: np array of the image (n, m , 1)
    :return: number of rows and columns
    """


def find_cell(cropped_dir_path, img_grid, img):
    # Find contours for image, which will detect all the boxes
    contours, hierarchy = cv2.findContours(
        img_grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort all the contours by top to bottom.
    (contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")

    idx = 0
    cells = []
    for c in boundingBoxes:
        # Returns the location and width,height for every contour
        x, y, w, h = c

        # If the cell height is greater than 10, width is >10,
        # and proportion is between 0.5 and 2
        if 2 > (w // h) > 0.5 and w > 10 and h > 10:
            idx += 1
            new_img = img[y:y + h, x:x + w]
            cells.append(new_img)
            data = text_extraction(new_img, "single_word")
            cv2.imwrite(f"{cropped_dir_path}{idx}_{data}.png", new_img)

    # For Debugging
    # Enable this line to see all contours.
    for i in range(len(contours)):
        tmp = img.copy()
        cv2.drawContours(tmp, contours, i, (0, 0, 255), 3)
        cv2.imwrite(f"Images/contour{i}.jpg", tmp)

    cv2.imwrite("img_contour.jpg", img)


if __name__ == '__main__':
    path = "test_image/img3.jpeg"
    is_handwritten = False

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

    r = np.logical_not(img_bin).astype(np.uint8) * 255 - img_grid
    # invert r
    r = 255 - r

    cv2.imwrite("Images/img_minus_grid.jpg", r)

    # find the contours
    find_cell("cells/", img_grid, r)