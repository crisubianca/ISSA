import cv2
import numpy as np


cam = cv2.VideoCapture('D:/Desktop/fii/anul III/Semestrul II/ISSA/Lab9/Lane Detection Test Video 01.mp4')

left_top_y = 0
left_top_x = 0
left_bottom_y = 0
left_bottom_x = 0
right_top_y = 0
right_top_x = 0
right_bottom_y = 0
right_bottom_x = 0
left_top_point = left_top_x, left_top_y
left_bottom_point = left_bottom_x, left_bottom_y
right_top_point = right_top_x, right_top_y
right_bottom_point = right_bottom_x, right_bottom_y


while True:

    ret, frame = cam.read()

    # ret (bool): Return code of the 'read' operation. Did we get an image or not?
    #             (if not maybe the camera is not detected/connected etc.)
    #
    # frame (array): The actual frame as an array.
    #                 Height x Width x 3 (3 colors, BGR) if color image.
    #                 Height x Width if Grayscale
    #                 Each element is 0-255.
    #                 You can slice it, reassign elements to change pixels, etc.

    if ret is False:
        break

    original_height = frame.shape[0] # 1280
    original_width = frame.shape[1] # 720

    height = int(original_height / 4)
    width = int(original_width / 4)

    frame = cv2.resize(frame, (width, height))
    # resize = cv2.resize(frame, (350, 200))
    # cv2.resize(frame, (new_width, new_height)) it returns the resized frame

    # To get the size of the frame use frame.shape (tuple of (height, width))


    cv2.imshow('1.Original', frame)
    # cv2.imshow(title_of_window, frame_array) displays an image


    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('2.Grayscale', grayscale)


    # points = (x,y) // tuples of 2 ints
    # the top is 0.0 and the bottom is 1.0, 0.55 is a bit lower than the half-point
    upper_left = (int(width * 0.46), int(height * 0.75))
    upper_right = (int(width * 0.54), int(height * 0.75))
    lower_left = (width * 0, height * 1)
    lower_right = (width * 1, height * 1)

    # placing the coordinates in a Numpy array in order:
    # np.array([pt1, pt2, pt3, pt4], dtype = np.int32)
    points_of_the_trapezoid = np.array([upper_right, upper_left, lower_left, lower_right], dtype=np.int32)

    # creating an empty black frame using np.zeros
    trapezoid = np.zeros((height, width), dtype=np.uint8)

    cv2.fillConvexPoly(trapezoid, points_of_the_trapezoid, 1)
    # cv2.fillConvexPoly(frame_in_which_to_draw, points_of_a_polygon,
    # color_to_draw_with) used to draw the trapezoid onto the frame

    # the points are a Numpy array
    #  the color is going to be 1 since we want an image of 1's and 0's

    # displaying only a white trapezoid we do :
    #  cv2.imshow(“Trapezoid”, trapezoid_frame * 255) since we have an array of
    # 0 and 1 the color 1 is VERY close to pure black (0), and 255 is pure white (and clearly
    # visible)
    cv2.imshow('3.Trapezoid', trapezoid*255)

    # we have an array of all 0’s and some 1’s in the shape of a trapezoid
    # displaying the trapezoid only with the street visible we
    # multiply each element in the grayscale frame with the corresponding
    # element in the trapezoid frame.
    road = trapezoid * grayscale
    cv2.imshow('4.Road', road)

    # we stretch the corners of the area of the screen you want to stretch (coord. of trapezoid)
    # and the corners of the area you want to stretch it to (the whole screen)

    # we already have the coord of the trapezoid corners (in trigonometrical order)
    # the coord of the screen corners are :
    # the upper left point is (0,0)
    # the upper right point is (frame_width, 0)
    # -||- for bottom corners

    # place them in trigonometrical order
    points_for_stretch = np.float32(
        np.array([(width * 1, height * 0), (width * 0, height * 0), lower_left, lower_right], dtype=np.int32))

    # we convert them to float32 since the stretching works with floats
    # trapezoid_bounds = np.float32(trapezoid_bounds)
    points_of_the_trapezoid = np.float32(points_of_the_trapezoid)

    # cv2.getPerspectiveTransform(bounds_of_current_area,
    #                             bounds_of_area_you_want_to_stretch_to)

    # the area we want to stretch it over is the entire frame
    # so its bounds are the corners of the screen
    magic_matrix = cv2.getPerspectiveTransform(points_of_the_trapezoid, points_for_stretch)

    # from getPerspectiveTransform you will get back a magical matrix that you can use to “stretch”
    # the trapezoid

    # this magical matrix will be used for the actual “stretching” using
    # cv2.warpPerspective(some_image, magic_matrix, (new_width, new_height))

    # warpPerspective should return the “stretched” image like in the example
    top_down = cv2.warpPerspective(road, magic_matrix, (width, height))
    cv2.imshow('5.Top-Down', top_down)


    # to blur an image we need to make each pixel to be the average value of its neighbors.
    # if we use more neighbors the blur will be stronger (more blurry), if we use less neighbors
    # the blur will be weaker (“clearer”)

    # we use a simple n x m matrix (for example 3 x 3). We move this matrix over the frame so that each pixel
    # will at some point be in its center (n and m must be odd numbers like 3, 5, 7, etc.) and that pixel become
    # the average of the elements in the matrix

    # use cv2.blur(frame, ksize = (n,n)) - this should return the blurred frame
    blur = cv2.blur(top_down, ksize=(7, 7))

    # ksize is a tuple with the dimensions of the blur area (also called a “kernel”).
    # usually ksize is a small square matrix like (3, 3), (5, 5), (7, 7)
    cv2.imshow('6.Blur', blur)

    sobel_vertical = np.float32([[-1, -2, -1],
                                 [0, 0, 0],
                                 [1, 2, 1]])

    sobel_horizontal = np.transpose(sobel_vertical)

    # cv2.filter2D(frame_as_float_32, -1, filter_matrix) applies a filter to a frame and returns
    # the result
    sobel_vertical = cv2.filter2D(np.float32(blur), -1, sobel_vertical)
    sobel_horizontal = cv2.filter2D(np.float32(blur), -1, sobel_horizontal)

    # 𝑚𝑎𝑡𝑟𝑖𝑥3 = √[(𝑚𝑎𝑡𝑟𝑖𝑥1)^2 + (𝑚𝑎𝑡𝑟𝑖𝑥2)^2]
    sobel = np.sqrt(sobel_vertical*sobel_vertical + sobel_horizontal*sobel_horizontal)

    # to convert the float32 matrices to uint8 matrices (so you can display them) without losing
    # quality you can use cv2.convertScaleAbs(my_matrix).

    # so to show your current images pass it trough cv2.convertScaleAbs(my_matrix). Think of
    # this function as a „toString()”.
    sobel = cv2.convertScaleAbs(sobel)

    cv2.imshow('7.Sobel', sobel)

    # each pixel below the threshold becomes absolute black, every pixel above the threshold becomes
    # absolute white
    returned_by_threshold, threshold = cv2.threshold(sobel, int(255 / 5), 255, cv2.THRESH_BINARY)

    cv2.imshow('8.Threshold', threshold)

    frame_copy = threshold.copy()

    nr_of_col = int(0.05 * frame_copy.shape[1])
    frame_copy[:, :nr_of_col] = 0
    frame_copy[:, -nr_of_col:] = 0

    indexes = np.argwhere(frame_copy > 1)
    midpoint = int(frame_copy.shape[1] / 2)
    left_indexes = indexes[indexes[:, 1] < midpoint]
    right_indexes = indexes[indexes[:, 1] >= midpoint]

    left_xs, left_ys = left_indexes[:, 1], left_indexes[:, 0]
    right_xs, right_ys = right_indexes[:, 1] - midpoint, right_indexes[:, 0]


    # finding the lines that detetct the edges of the lane
    left_line = np.polynomial.polynomial.polyfit(left_xs, left_ys, deg=1)
    right_line = np.polynomial.polynomial.polyfit(right_xs, right_ys, deg=1)

    b = left_line[0]
    a = left_line[1]
    left_top_y = height
    left_top_x = (height - b) / a
    left_bottom_y = 0
    left_bottom_x = -b / a

    d = right_line[0]
    c = right_line[1]
    right_top_y = height
    right_top_x = (height - d) / c + int(width / 2)
    right_bottom_y = 0
    right_bottom_x = -d / c + int(width / 2)

    if int(width / 2) >= left_top_x >= 0 and int(width / 2) >= left_bottom_x >= 0:
        left_top_point = int(left_top_x), int(left_top_y)
        left_bottom_point = int(left_bottom_x), int(left_bottom_y)

    if width >= right_top_x >= int(width / 2) and width >= right_bottom_x >= int(width / 2):
        right_top_point = int(right_top_x), int(right_top_y)
        right_bottom_point = int(right_bottom_x), int(right_bottom_y)


    lines = cv2.line(frame_copy, left_bottom_point, left_top_point, (200, 0, 0), 5)
    lines = cv2.line(lines, right_bottom_point, right_top_point, (100, 0, 0), 5)


    cv2.imshow('9.Lines', lines)

    blank = np.zeros((height, width), dtype=np.uint8)
    cv2.line(blank, left_top_point, left_bottom_point, (255, 0, 0), 3)

    magic_matrix = cv2.getPerspectiveTransform(points_for_stretch, points_of_the_trapezoid)
    final_left = cv2.warpPerspective(blank, magic_matrix, (width, height))

    cv2.imshow('10.Final Left', final_left)


    left_lane = np.argwhere(final_left > 1)
    left_xs, left_ys = left_lane[:, 1], left_lane[:, 0]

    blank = np.zeros((height, width), dtype=np.uint8)
    cv2.line(blank, right_top_point, right_bottom_point, (255, 0, 0), 3)

    magic_matrix = cv2.getPerspectiveTransform(points_for_stretch, points_of_the_trapezoid)
    final_right = cv2.warpPerspective(blank, magic_matrix, (width, height))

    cv2.imshow('11.Final Right', final_right)

    right_lane = np.argwhere(final_right > 1)
    right_xs, right_ys = right_lane[:, 1], right_lane[:, 0]

    final = frame.copy()
    final[final_left > 0] = (50, 50, 250)
    final[final_right > 0] = (50, 250, 50)

    cv2.imshow('12.Final', final)

    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
    # cv2.waitKey(n) waits n ms for a key to be pressed and returns the code of that key.
    # cv2.waitKey(n) & 0xFF gives the ASCII code of the letter (so the if is executed when we press
    # “q”).

cam.release()
cv2.destroyAllWindows()