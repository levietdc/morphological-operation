import sys
import getopt
import cv2
import numpy as np
from morphological_operator import binary
from morphological_operator import gray


def operator(in_file, out_file, mor_op, wait_key_time=0):
    img_origin = cv2.imread(in_file)
    cv2.imshow('original image', img_origin)
    cv2.waitKey(wait_key_time)

    img_gray = cv2.imread(in_file, 0)
    cv2.imshow('gray image', img_gray)
    cv2.waitKey(wait_key_time)

    (thresh, img) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow('binary image', img)
    cv2.waitKey(wait_key_time)

    kernel = np.ones((3, 3), np.uint8)
    kernel_hit_miss = np.array([[0, 1, 0], [1, -1, 1], [0, 1, 0]])
    img_out = None

    '''
    TODO: implement morphological operators
    '''
    if mor_op == 'dilate_bin':
        img_dilation = cv2.dilate(img, kernel)
        cv2.imshow('OpenCV dilation image', img_dilation)
        cv2.waitKey(wait_key_time)

        img_dilation_manual = binary.dilate(img, kernel)
        cv2.imshow('manual dilation image', img_dilation_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_dilation
    elif mor_op == 'erode_bin':
        img_erosion = cv2.erode(img, kernel)
        cv2.imshow('OpenCV erosion image', img_erosion)
        cv2.waitKey(wait_key_time)

        img_erosion_manual = binary.erode(img, kernel)
        cv2.imshow('manual erosion image', img_erosion_manual)
        cv2.waitKey(wait_key_time)
        img_out = img_erosion_manual
    elif mor_op == 'closing_bin':
        img_closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('OpenCV closing image', img_closing)
        cv2.waitKey(wait_key_time)

        img_closing_manual = binary.closing(img, kernel)
        cv2.imshow('manual closing image', img_closing_manual)
        cv2.waitKey(wait_key_time)
    elif mor_op == 'opening_bin':
        img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        cv2.imshow('OpenCV opening image', img_opening)
        cv2.waitKey(wait_key_time)

        img_opening_manual = binary.opening(img, kernel)
        cv2.imshow('manual opening image', img_opening_manual)
        cv2.waitKey(wait_key_time)
    elif mor_op == 'hitmiss_bin':
        img_hit_miss = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel_hit_miss)
        cv2.imshow('OpenCV hit or miss image', img_hit_miss)
        cv2.waitKey(wait_key_time)

        img_hitmiss_manual = binary.hit_or_miss(img, kernel_hit_miss)
        cv2.imshow('manual hit or miss image', img_hitmiss_manual)
        cv2.waitKey(wait_key_time)
    elif mor_op == 'thinning_bin':
        img_thinning = cv2.ximgproc.thinning(img)
        cv2.imshow('OpenCV thinning image', img_thinning)
        cv2.waitKey(wait_key_time)

        thinning_manual = binary.thinning(img)
        cv2.imshow('manual thinning image', thinning_manual)
        cv2.waitKey(wait_key_time)

    elif mor_op == 'boundary_bin':
        img_boundary = img - cv2.erode(img, kernel)
        cv2.imshow('OpenCV boundary extraction image', img_boundary)
        cv2.waitKey(wait_key_time)

        img_boundary_manual = binary.extract_boundary(img, kernel)
        cv2.imshow('manual boundary extraction image', img_boundary_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_boundary_manual

    # Grayscale image

    elif mor_op == 'erode_gray':
        img_erosion = cv2.morphologyEx(img_gray, cv2.MORPH_ERODE, kernel)
        cv2.imshow('OpenCV erosion image', img_erosion)
        cv2.waitKey(wait_key_time)

        img_erosion_manual = gray.erode(img_gray, kernel)
        cv2.imshow('manual ersion image', img_erosion_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_erosion_manual

    elif mor_op == 'dilate_gray':
        img_dilation = cv2.morphologyEx(img_gray, cv2.MORPH_DILATE, kernel)
        cv2.imshow('OpenCV dilation image', img_dilation)
        cv2.waitKey(wait_key_time)

        img_dilation_manual = gray.dilate(img_gray, kernel)
        cv2.imshow('manual dilation image', img_dilation_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_dilation_manual

    elif mor_op == 'open_gray':
        img_opening = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)
        cv2.imshow('OpenCV opening image', img_opening)
        cv2.waitKey(wait_key_time)

        img_opening_manual = gray.open(img_gray, kernel)
        cv2.imshow('manual opening image', img_opening_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_opening_manual

    elif mor_op == 'close_gray':

        img_closing = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('OpenCV closeing image', img_closing)
        cv2.waitKey(wait_key_time)

        img_closing_manual = gray.close(img_gray, kernel)
        cv2.imshow('manual closing image', img_closing_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_closing_manual
    elif mor_op == 'gradient_gray':
        img_gradient = cv2.morphologyEx(img_gray, cv2.MORPH_GRADIENT, kernel)
        cv2.imshow('OpenCV gradient image', img_gradient)
        cv2.waitKey(wait_key_time)

        img_gradient_manual = gray.gradient(img_gray, kernel)
        cv2.imshow('manual gradient image', img_gradient_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_gradient_manual

    elif mor_op == 'tophat_gray':
        img_top_hat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)
        cv2.imshow('OpenCV top-hat image', img_top_hat)
        cv2.waitKey(wait_key_time)

        img_top_hat_manual = gray.top_hat(img_gray, kernel)
        cv2.imshow('manual top-hat image', img_top_hat_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_top_hat_manual

    elif mor_op == 'blackhat_gray':
        img_black_hat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)
        cv2.imshow('OpenCV black-hat image', img_black_hat)
        cv2.waitKey(wait_key_time)

        img_black_hat_manual = gray.black_hat(img_gray, kernel)
        cv2.imshow('manual black-hat image', img_black_hat_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_black_hat_manual


def main(argv):
    input_file = ''
    output_file = ''
    mor_op = ''
    wait_key_time = 0

    description = 'main.py -i <input_file> -o <output_file> -p <mor_operator> -t <wait_key_time>'

    try:
        opts, args = getopt.getopt(argv, "hi:o:p:t:", ["in_file=", "out_file=", "mor_operator=", "wait_key_time="])
    except getopt.GetoptError:
        print(description)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(description)
            sys.exit()
        elif opt in ("-i", "--in_file"):
            input_file = arg
        elif opt in ("-o", "--out_file"):
            output_file = arg
        elif opt in ("-p", "--mor_operator"):
            mor_op = arg
        elif opt in ("-t", "--wait_key_time"):
            wait_key_time = int(arg)

    print('Input file is ', input_file)
    print('Output file is ', output_file)
    print('Morphological operator is ', mor_op)
    print('Wait key time is ', wait_key_time)

    operator(input_file, output_file, mor_op, wait_key_time)
    cv2.waitKey(wait_key_time)


if __name__ == "__main__":
    main(sys.argv[1:])
