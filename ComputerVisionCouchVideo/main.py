


# importing the module
import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt
from PIL import Image
import textwrap






# reading the video
from urllib3.filepost import writer

source = cv2.VideoCapture('Couch_trimed1.mp4') #myVideo.mp4

length = int(source.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = source.get(cv2.CAP_PROP_FPS)

print("The width and height are: ", width, "x", height)

print("The number of frames of the video are: ", length)

# We need to set resolutions.
# so, convert them from float to integer.
frame_width = int(source.get(3))
frame_height = int(source.get(4))

size = (frame_width, frame_height)

result = cv2.VideoWriter('result.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size, 1)




i = 1
# running the loop
while i < 601:  # was 120

    # extracting the frames
    ret, img = source.read()

    # converting to gray-scale
    if ((i >= 1 and i < 10) or (i >= 20 and i < 30)):  # i >= 1 and i < 10) or (i >= 20 and i < 30
        newimg = img

        img = cv2.putText(
            img,
            text="Here you see a colored video.",
            org=(10, 1800),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 255),
            thickness=3
        )
    elif ((i >= 10 and i < 20) or (i > 30 and i < 45)):  # ((i >= 10 and i < 20) or (i > 30 and i < 40))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        newimg = cv2.merge([gray, gray, gray])

        newimg = cv2.putText(
            newimg,
            text="And here a gray scaled video.",
            org=(10, 1800),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 255),
            thickness=3
        )



    elif (i >= 45 and i < 65):  # (i >= 40 and i < 50)


        newimg = cv2.GaussianBlur(img, (51, 51), 0)  # standard deviation in the x direction

        newimg = cv2.putText(
            newimg,
            text="Here is a blured image using Gaussian Blur filter of 51 x 51.",
            org=(10, 1800),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 255),
            thickness=3
        )

    elif (i >= 65 and i < 85):  # (i >= 40 and i < 50)


        newimg = cv2.GaussianBlur(img, (7, 7), 0)  # standard deviation in the x direction

        newimg = cv2.putText(
            newimg,
            text="And here is we use the same filter but 6 x 6.",
            org=(10, 1800),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 255),
            thickness=3
        )

    elif (i >= 85 and i <= 105):  # (i >= 50 and i <= 60)



        newimg = cv2.bilateralFilter(img, 9, 75, 75)
        newimg = cv2.putText(
            newimg,
            text="And here is a Blured image using Bilateral Filter.",
            org=(10, 1800),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 255),
            thickness=3
        )


    elif (i >= 105 and i <= 125):  # (i >= 50 and i <= 60)


        newimg = cv2.bilateralFilter(img, 22, 75, 75) #img, 22, 75, 75
        newimg = cv2.putText(
            newimg,
            text="Diameter of each pixel neighborhood in now 22 (was 9 earlier).",
            org=(10, 1800),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 255),
            thickness=3
        )

    elif (i > 125 and i <= 145):  # (i > 60 and i <= 80)

        # kernel is a square or a shape which we want to apply to the image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        newimg = cv2.merge([gray, gray, gray])
        # median = cv2.medianBlur(img, 3)
        ret, newimg = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        newimg = cv2.merge((newimg.copy(), newimg.copy(), newimg.copy()))
        # newimg = cv2.merge([b,b,b])

        # frame_HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # newimg = cv2.inRange(frame_HSV, (325, 13, 98), (318, 55, 86))
        newimg = cv2.putText(
            newimg,
            text="Here we apply Thresholding in the keyboard - Binary.",
            org=(10, 1800),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 255),
            thickness=3
        )



    elif (i > 145 and i <= 200):  # (i > 60 and i <= 80)

        # kernel is a square or a shape which we want to apply to the image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        newimg = cv2.merge([gray, gray, gray])
        ret, newimg = cv2.threshold(img, 80, 122, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        newimg = cv2.merge((newimg.copy(), newimg.copy(), newimg.copy()))

        newimg = cv2.putText(
            newimg,
            text="Same thresholding but with different pixel values (not suitable).",
            org=(10, 1800),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 255),
            thickness=3
        )

    elif (i > 200 and i <= 215):  # (i > 60 and i <= 80)
        newimg = cv2.Sobel(src=img, ddepth=cv2.CV_8U, dx=1, dy=0, ksize=5)

        newimg = cv2.putText(
            newimg,
            text="Sobel applied on the x axis.",
            org=(10, 1800),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.5,
            color=(255, 255, 255),
            thickness=3
        )


    elif (i > 215 and i <= 230):  # (i > 60 and i <= 80)
        newimg = cv2.Sobel(src=img, ddepth=cv2.CV_8U, dx=0, dy=1, ksize=5)

        newimg = cv2.putText(
            newimg,
            text="Sobel filter applied on the y axis.",
            org=(10, 1800),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.5,
            color=(255, 255, 255),
            thickness=3
        )

    elif (i > 230 and i <= 250):  # (i > 60 and i <= 80)
        newimg = cv2.Sobel(src=img, ddepth=cv2.CV_8U, dx=1, dy=1, ksize=5)

        newimg = cv2.putText(
            newimg,
            text="Here you see Combined Sobel and clearly the edges.",
            org=(10, 1800),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 255),
            thickness=3
        )

    elif (i > 250 and i <= 270):  # (i > 60 and i <= 80)
        newimg = img
        newimg2 = img.copy()
        newimg2 = cv2.cvtColor(newimg2, cv2.COLOR_BGR2GRAY)

        newimg2 = cv2.medianBlur(newimg2, 5)

        rows = newimg2.shape[0]
        # can detect sometime the circular in stabilo
        circles = cv2.HoughCircles(newimg2, cv2.HOUGH_GRADIENT, 2, rows / 8,
                                   param1=110, param2=25,
                                   minRadius=1, maxRadius=15)




        if circles is not None:
            circles = np.uint16(np.around(circles))
            for j in circles[0, :]:
                center = (j[0], j[1])
                cv2.circle(newimg, center, 1, (0, 0, 255), 1)  # circle center
                radius = j[2]  # circle outline
                cv2.circle(newimg, center, radius, (0, 0, 255), 2)

        #cv2.imshow("my test", img)

        newimg = cv2.putText(
            newimg,
            text="Trying to detect very small circles using Hough transformation.",
            org=(10, 1800),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 255),
            thickness=3
        )



    elif (i > 270 and i <= 290):  # (i > 60 and i <= 80)
        newimg = img
        newimg2 = img.copy()
        newimg2 = cv2.cvtColor(newimg2, cv2.COLOR_BGR2GRAY)

        newimg2 = cv2.medianBlur(newimg2, 5)

        rows = newimg2.shape[0]
        circles = cv2.HoughCircles(newimg2, cv2.HOUGH_GRADIENT, 2, rows / 8,
                                   param1=120, param2=30,
                                   minRadius=1, maxRadius=45)


        if circles is not None:
            circles = np.uint16(np.around(circles))
            for j in circles[0, :]:
                center = (j[0], j[1])
                cv2.circle(newimg, center, 1, (0, 0, 255), 1)  # circle center
                radius = j[2]  # circle outline
                cv2.circle(newimg, center, radius, (0, 0, 255), 2)

        #cv2.imshow("my test", img)

        newimg = cv2.putText(
            newimg,
            text="Trying to detect small circles using Hough transformation.",
            org=(10, 1800),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 255),
            thickness=3
        )




    elif (i > 290 and i <= 320):  # (i > 60 and i <= 80)
        newimg = img
        newimg2 = img.copy()
        newimg2 = cv2.cvtColor(newimg2, cv2.COLOR_BGR2GRAY)

        newimg2 = cv2.medianBlur(newimg2, 5)

        rows = newimg2.shape[0]
        circles = cv2.HoughCircles(newimg2, cv2.HOUGH_GRADIENT, 2, rows / 8,
                                   param1=125, param2=30,
                                   minRadius=4, maxRadius=90)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for j in circles[0, :]:
                center = (j[0], j[1])
                cv2.circle(newimg, center, 1, (0, 0, 255), 1)  # circle center
                radius = j[2]  # circle outline
                cv2.circle(newimg, center, radius, (0, 0, 255), 2)

        # cv2.imshow("my test", img)

        newimg = cv2.putText(
            newimg,
            text="Trying to detect bigger circles using Hough transformation.",
            org=(10, 1800),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 255),
            thickness=3
        )

    elif (i > 320 and i <= 350):  # (i > 60 and i <= 80)
        newimg = img
        newimg2 = img.copy()
        newimg2 = cv2.cvtColor(newimg2, cv2.COLOR_BGR2GRAY)

        newimg2 = cv2.medianBlur(newimg2, 5)

        rows = newimg2.shape[0]
        circles = cv2.HoughCircles(newimg2, cv2.HOUGH_GRADIENT, 1, rows / 8,
                                   param1=200, param2=30,
                                   minRadius=40, maxRadius=160)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for j in circles[0, :]:
                center = (j[0], j[1])
                cv2.circle(newimg, center, 1, (0, 0, 255), 5)  # circle center
                radius = j[2]  # circle outline
                cv2.circle(newimg, center, radius, (0, 0, 255), 3)

        #cv2.imshow("my test", img)

        newimg = cv2.putText(
            newimg,
            text="Circular object of interest detected successfully!!",
            org=(10, 1800),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 255),
            thickness=3
        )




    elif (i > 350 and i <= 370):

        newimg = img
        newimg2 = img.copy()
        newimg2 = cv2.cvtColor(newimg2, cv2.COLOR_BGR2GRAY)

        newimg2 = cv2.medianBlur(newimg2, 5)

        rows = newimg2.shape[0]
        circles = cv2.HoughCircles(newimg2, cv2.HOUGH_GRADIENT, 1, rows / 8,
                                   param1=200, param2=30,
                                   minRadius=40, maxRadius=160)


        if circles is not None:
            circles = np.uint16(np.around(circles))
            #print("x, y, r", circles)
            #[[[x, y, r]]] = circles
            for j in circles[0, :]:
                [x, y, r] = j
                center = (j[0], j[1])
                # circle center
                cv2.circle(newimg, center, 1, (255, 0, 0), 5)
                # circle outline
                radius = j[2]
                cv2.circle(newimg, center, radius, (255, 0, 0), 3)
                cv2.rectangle(newimg, (x-r,y-r), (x+r, y+r), (0, 0, 255), 3)
        # cv2.imshow("my test", img)
        newimg = cv2.putText(
            newimg,
            text="Detect object with Hough transform. and a rectangle around it!",
            org=(10, 1800),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 255),
            thickness=3
        )



    elif (i > 370 and i <= 400):

        newimg = img
        newimg2 = img.copy()
        newimg2 = cv2.cvtColor(newimg2, cv2.COLOR_BGR2GRAY)

        newimg2 = cv2.medianBlur(newimg2, 5)

        rows = newimg2.shape[0]
        circles = cv2.HoughCircles(newimg2, cv2.HOUGH_GRADIENT, 1, rows / 8,
                                   param1=200, param2=30,
                                   minRadius=40, maxRadius=160)


        if circles is not None:
            circles = np.uint16(np.around(circles))
            #print("x, y, r", circles)
            #[[[x, y, r]]] = circles
            for j in circles[0, :]:
                [x, y, r] = j
                center = (j[0], j[1])
                # circle center
                cv2.circle(newimg, center, 1, (255, 0, 0), 5)
                # circle outline
                radius = j[2]
                cv2.circle(newimg, center, radius, (255, 0, 0), 3)
                cv2.rectangle(newimg, (x-r,y-r), (x+r, y+r), (0, 0, 255), 3)
        # cv2.imshow("my test", img)
        newimg = cv2.putText(
            newimg,
            text="Detect object with Hough transformation.",
            org=(10, 1800),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 255),
            thickness=3
        )


    elif (i > 400 and i <= 450):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        newimg = cv2.merge([gray, gray, gray])

        newimg = cv2.putText(
            newimg,
            text="Here we make the video gray again.",
            org=(10, 1800),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 255),
            thickness=3
        )




    elif (i > 450 and i <= 500):  # (i > 60 and i <= 80)
        newimg = cv2.Sobel(src=img, ddepth=cv2.CV_8U, dx=1, dy=1, ksize=5)

        newimg = cv2.putText(
            newimg,
            text="Here we use Combined Sobel.",
            org=(50, 1800),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 255),
            thickness=3
        )


    elif (i > 500 and i <= 600):

        newimg = img
        newimg2 = img.copy()
        newimg2 = cv2.cvtColor(newimg2, cv2.COLOR_BGR2GRAY)

        newimg2 = cv2.medianBlur(newimg2, 5)

        rows = newimg2.shape[0]
        circles = cv2.HoughCircles(newimg2, cv2.HOUGH_GRADIENT, 1, rows / 8,
                                   param1=200, param2=30,
                                   minRadius=40, maxRadius=160)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            # print("x, y, r", circles)
            # [[[x, y, r]]] = circles
            for j in circles[0, :]:
                [x, y, r] = j
                center = (j[0], j[1])
                # circle center
                cv2.circle(newimg, center, 1, (255, 0, 0), 5)
                # circle outline
                radius = j[2]
                cv2.circle(newimg, center, radius, (255, 0, 0), 3)
                cv2.rectangle(newimg, (x - r, y - r), (x + r, y + r), (0, 0, 255), 3)
        # cv2.imshow("my test", img)
        newimg = cv2.putText(
            newimg,
            text="Here we keep detecting our object of interest using Hough transf.",
            org=(10, 1800),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 255),
            thickness=3
        )



    result.write(newimg)

    # displaying the video
    cv2.imshow("Live", newimg)

    # exiting the loop
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    i = i + 1


