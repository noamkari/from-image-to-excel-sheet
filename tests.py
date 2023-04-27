from main import *
import pandas as pd


def test_1():
    # read the image
    img = read_image("test_image/img_1.png")

    # find the cells in the image
    cells = find_cells(img)

    # find the numbers in the cells
    data = detect_text(cells)

    assert data == ['6', '0', '6', '4', '3', '0', '4', '8', '0', '0', '0', '3']

    # insert the data to Excel file
    insert_to_excel(data, "test_image/test_1.xlsx")

    # open expected output file as dataframes
    expected = pd.read_excel('test_image/img_1_exp.xlsx')
    actual = pd.read_excel('test_image/test_1.xlsx')

    # compare the dataframes
    assert expected.equals(actual)


def test_4():
    # read the image
    img = read_image("test_image/img_4.png")

    # find the cells in the image
    cells = find_cells(img)

    # find the data in the cells
    data = detect_text(cells)

    assert data == ['Total', 10, 8, 6, 4, 3, 3, 3, 3, 3, 2]

    # insert the data to Excel file
    insert_to_excel(data, "test_image/test_4.xlsx")

    # open expected output file as dataframes
    expected = pd.read_excel('test_image/img_4_exp.xlsx')
    actual = pd.read_excel('test_image/test_4.xlsx')

    # compare the dataframes
    assert expected.equals(actual)
