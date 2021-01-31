import numpy as np
import argparse
import cv2
import imutils

# arguments to take from terminal
parser = argparse.ArgumentParser(
    description="This is a scanner for papers meaning each image of a paper that is entered the script will output"
                " a binary scanned image.\n"
                "just enter an image path of rectangle shaped paper and "
                "a path for where the final image to be saved at...",
    epilog="And that's it :) ... ")
parser.add_argument('o_image', type=str, metavar='<path of an image to be scanned>',
                    help="please enter a picture path and the picture containing a rectangle shaped paper")
parser.add_argument('final_p', type=str, metavar='<the path you want to save the scanned image at>',
                    help="please enter a path for the scanned picture to be saved at or just the"
                         " name of the file and it will be saved in the working directory"
                         ".")
args = vars(parser.parse_args())


def order_point(points):
    """
    Ordering the points in the shape of (4,2) where top-left corner should be in index 0,
    top-right at index 1 , bottom-right at index 2 and bottom left at index 3 ..
    :arg points: points that represent the four corners of the paper in an image
    :returns ordered_cor: an Ordered points 2D array [[top-left] [top-right] [bottom-right] [bottom-left]]
    """

    # initialize a list of coordinates that will be ordered such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the bottom-right, and the fourth is the bottom-left

    ordered_cor = np.zeros((4, 2), dtype=np.float32)
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = points.sum(axis=1)
    ordered_cor[0] = points[np.argmin(s)]
    ordered_cor[2] = points[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(points, axis=1)
    ordered_cor[1] = points[np.argmin(diff)]
    ordered_cor[3] = points[np.argmax(diff)]
    # return the ordered coordinates
    return ordered_cor


def detect_corners(origin_img):
    """
    In this function we will detect the four corners of the given image ,

    :param origin_img: the image we want to scan
    :returns: a list of the four corners in the given image
    """
    print("----- detecting the image corners -----")
    # saving the ratio of the image because we want to resize it and work with the resized image for computational
    # efficiency and then we return
    # the four corners of the image with the original size
    original_ratio = origin_img.shape[0] / 1000.0
    origin_img = imutils.resize(origin_img, height=1000)

    # convert the image to grayscale, blur it, and find edges using canny edge detector
    # in the image
    gray_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    edged_img = cv2.Canny(blurred_img, 75, 200)

    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the contour
    cnts = cv2.findContours(edged_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0]
    # taking top five contours by area but we eventaully we'll need just 4
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    # loop over the contours
    for c in cnts:
        # approximate the contour
        epsilon = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * epsilon, True)
        # if our approximated contour has four points, then we
        # can assume that we have found our ROI (region of interest)
        if len(approx) == 4:
            roi_Cnt = approx
            break

    print("----- found four corners -----")
    return roi_Cnt.reshape(4, 2) * original_ratio


def transformation(origin_img, points):
    """
    Taking an image that contains a paper and then we use prospective transform to get that top to bottom picture
    :param origin_img: The original image containing a paper
    :param points:the four points of the corners of the paper in the image
    :returns: a wrapped image of the region of interest which is the paper in the image
    """
    print("----- transforming the image -----")
    (tl, tr, br, bl) = order_point(points=points)

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left x-coordinates or the top-right and top-left x-coordinates
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_a))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "top down view",
    # the order in dst is top-left, top-right, bottom-right, and bottom-left
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(points, dst)
    wrapped = cv2.warpPerspective(origin_img, M, (max_width, max_height))

    print("----- image is transformed -----")
    # returning the warped image
    return wrapped


def scanned(wrapped_img, save_path):
    """
    An image of a paper and then turning into a binary image and saving the binary image
    and basically our given image containing a paper is now scanned
    :param wrapped_img: an image that is supposed to be the paper part of the original given image
    :param save_path: the path we want our scanned image to be saved at
    :return:None
    """
    # to gray scale
    gray_img = cv2.cvtColor(wrapped_img, cv2.COLOR_BGR2GRAY)
    # values need to be from 0-255
    gray_img = gray_img.astype(dtype=np.uint8)
    # thresh using OTSU
    _, thresh_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

    # our final result needed to mirror flipped
    flipped_img = cv2.flip(thresh_img, 1)

    # because the images need to be rotated in my case , this is for the two specific image given in the dataset
    if "Game" in args["o_image"]:
        flipped_img = cv2.rotate(flipped_img, cv2.ROTATE_90_CLOCKWISE)
    elif "page" in args["o_image"]:
        for _ in range(3):
            flipped_img = cv2.rotate(flipped_img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(save_path, flipped_img)

    print(f"----- done scanning the image and it is saved in the given path => {save_path} -----")


if __name__ == "__main__":
    if args["o_image"] and args["final_p"]:
        img = cv2.imread(args["o_image"])
        four_corners = detect_corners(origin_img=img)
        wrapped = transformation(origin_img=img, points=four_corners.astype(dtype="float32"))
        scanned(wrapped_img=wrapped, save_path=args["final_p"])
