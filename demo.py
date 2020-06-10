import cv2
import numpy as np
import os
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from PIL import ImageEnhance
import math

line_size = 4

# Get the version, since the return value in opencv3 and opencv4 are different

version = cv2.__version__.split(".")[0]

# extract_s = cv2.xfeatures2d.SIFT_create(100) # use SIFT Feature
extract_s = cv2.xfeatures2d.SURF_create(400)  # SURF

# Use FLANN for matching
FLANN_INDEX_KDTREE = 0
indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=20)
searchParams = dict(checks=50)
flann = cv2.FlannBasedMatcher(indexParams, searchParams)

database = {"red": dict(),
            "black":dict(),
            }


def cos_dist(a, b):
    if len(a) != len(b):
        return None
    part_up = 0.0
    a_sq = 0.0
    b_sq = 0.0
    for a1, b1 in zip(a, b):
        part_up += a1 * b1
        a_sq += a1 ** 2
        b_sq += b1 ** 2
    part_down = math.sqrt(a_sq * b_sq)
    if part_down == 0.0:
        return None
    else:
        return part_up / part_down


# this function is confined to rectangle
def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")


def order_points_quadrangle(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left and bottom-left coordinate, use it as an
    # base vector to calculate the angles between the other two vectors

    vector_0 = np.array(bl - tl)
    vector_1 = np.array(rightMost[0] - tl)
    vector_2 = np.array(rightMost[1] - tl)

    angle = [np.arccos(cos_dist(vector_0, vector_1)), np.arccos(cos_dist(vector_0, vector_2))]
    (br, tr) = rightMost[np.argsort(angle), :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")


def get_4_point(image):
    hsv_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    # get the mask of the blue labels
    low_hsv1 = np.array([110, 100, 100])
    high_hsv1 = np.array([124, 255, 255])
    blue_mask1 = cv2.inRange(hsv_image, lowerb=low_hsv1, upperb=high_hsv1)

    if version=="3": #opencv 3
        _, contours, _ = cv2.findContours(blue_mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    elif version=="4": #opencv 4
        contours, _ = cv2.findContours(blue_mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    else:
        print("Unkown opencv version:",version)
        exit()

    new_contours =[]
    for c in contours:
        if cv2.contourArea(c)>1000:
            new_contours.append(c)

    if len(new_contours)!=4:
        print("Image processing failed")
        exit()

    draw_point_list = []
    # get circle contours
    for c in new_contours:
        (x, y), apple_radius = cv2.minEnclosingCircle(c)
        draw_point_list.append([x,y])

    #Arrange the detected 4 blue dots in the order of upper left, upper right, lower right, lower left
    draw_point_list = order_points_quadrangle(np.array(draw_point_list))

    return draw_point_list


def enhance(image,factor=2 ):
    """
    Improve image contrast
    """
    gray_image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if factor==1:
        return gray_image
    pil_img = Image.fromarray(gray_image)
    im_contrast = ImageEnhance.Contrast(pil_img)
    im_contrast.enhance(factor)

    image = np.asarray(pil_img)

    return image


def cv_imread(file_path):
    # Prevent unreadable path
    if not os.path.exists(file_path):
        raise Exception("invalid:{}".format(file_path))
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img

def cv_imwrite(imgpath,img):
    cv2.imencode('.jpg', img)[1].tofile(imgpath)

def load_database(database_root_dir):
    """
    Load the pictures
    """
    read_database_dir = os.path.join(database_root_dir,"red")

    for i in os.listdir(read_database_dir):
        chess_image = cv_imread(os.path.join(read_database_dir,i))
        chess_image = enhance(chess_image)

        database["red"][i.split(".")[0]] = extract_s.detectAndCompute(chess_image, None) # get the features


    black_database_dir = os.path.join(database_root_dir, "black")
    for i in os.listdir(black_database_dir):
        chess_image = cv_imread(os.path.join(black_database_dir,i))
        chess_image = enhance(chess_image)
        database["black"][i.split(".")[0]] = extract_s.detectAndCompute(chess_image, None)# get the features



def preproess_image(src_image):
    """
    The original input image is too larges, andd the actual processing does not requires such a clear picture
    
    """
    min_dst = 1000.0
    image_height,image_width = src_image.shape[:2]
    if image_height <image_width:
        scalar = min_dst/image_height

    else:
        scalar = min_dst/image_width

    new_image_height = int(image_height*scalar)
    new_image_width  = int(image_width*scalar)

    new_image = cv2.resize(src_image,(new_image_width,new_image_height))
    return new_image


def get_color(image):
    """
    Classify the chess pieces by the color
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    low_hsv1 = np.array([0, 50, 5])
    high_hsv1 = np.array([8, 255, 255])
    red_mask1 = cv2.inRange(hsv, lowerb=low_hsv1, upperb=high_hsv1)

    low_hsv2 = np.array([156, 20, 5])
    high_hsv2 = np.array([180, 255, 255])
    red_mask2 = cv2.inRange(hsv, lowerb=low_hsv2, upperb=high_hsv2)

    red_mask = cv2.bitwise_or(red_mask1,red_mask2)

    H,W = image.shape[:2]

    if cv2.countNonZero(red_mask) > (H*W)/10 :
        return "red"
    else:
        return "black"


exist_name = []

def get_name(name):
    times = 0
    name = name.split("_")[0]
    for i in exist_name:
        if i == name:
            times+=1
    exist_name.append(name)
    return name+"_"+str(times)

def match(chess_image,color ):
    
    image = enhance(chess_image)

    kp1, des1 = extract_s.detectAndCompute(image, None)
    names = []
    good_nums = []
    for chess_name in database[color]:
        kp, des = database[color][chess_name]

        matches = flann.knnMatch(des1, des, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if len(good)>0:
            names.append(chess_name)
            good_nums.append(len(good))
    if len(good_nums):
        match_index = good_nums.index(max(good_nums))
        #cv_imwrite(os.path.join("ff",get_name(names[match_index])+".jpg"),chess_image)
            #Save the crop of each chess in the trial process
        return names[match_index]
    else:
        return None



def crop_chess(image,circle):
    x1,y1,x2,y2 = circle[0] - circle[2],\
                  circle[1] - circle[2],\
                  circle[0] + circle[2],\
                  circle[1] + circle[2]
    r = circle[2]
    x1, y1, x2, y2 = int(x1-r/5.0),int(y1-r/5.0),int(x2+r/5.0),int(y2+r/5.0) # keeps the background aroud the edge

    if x1<0:x1 =0

    if y1<0: y1=0

    if x2>image.shape[1]:
        x2 = image.shape[1]

    if y2>image.shape[0]:
        y2 = image.shape[0]


    chess_image = image[y1:y2,x1:x2]

    chess_image = cv2.resize(chess_image, (100, 100),interpolation=cv2.INTER_AREA)

    color = get_color(chess_image)

    match_name = match(chess_image,color)

    return match_name


if __name__ == '__main__':

    database_root_dir = "data_base"  # root library

    test_image_dir = "images_2"  # images for testing

    test_image_out_dir = "images_2_out"  # output

    load_database(database_root_dir)

    if not os.path.exists(test_image_out_dir):
        os.makedirs(test_image_out_dir)

    cv2.namedWindow("show", 0)

    detect_count =0 #count the number for recognization

    for img_name in os.listdir(test_image_dir):
        image_path = os.path.join(test_image_dir,img_name)

        base_image = cv2.imread(image_path)


        draw_point_list = get_4_point(base_image)

        cv2.imshow("show",base_image)

        point_list = np.reshape(draw_point_list,newshape=(4,2)).astype(np.float32)

        H_rows, W_cols= base_image.shape[:2]

        pts2 = np.float32([[0, 0],[1000,0],[1000,1000],[0, 1000]])

        # Generate rectify matrix; perform perspective transformation to correct image
        M = cv2.getPerspectiveTransform(point_list, pts2)
        dst = cv2.warpPerspective(base_image, M, (1000,1000))

        im_gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)

        #Find the circle on the chessboard, which are the positions of chess pieces
        circles = cv2.HoughCircles(im_gray, cv2.HOUGH_GRADIENT, 1.5, 65  , param1=100, param2=50, minRadius=30, maxRadius=40 )

        # Write on the image
        pil_image = Image.fromarray(dst.copy())

        font = ImageFont.truetype("simhei.ttf",30)

        draw = ImageDraw.Draw(pil_image)

        for i in circles[0]:
            match_name = crop_chess(dst,i)

            if  match_name is not  None:
                detect_count+=1
                draw_color = (0, 0, 255)
                if match_name.startswith("黑"):
                    draw_color = (0, 0, 0)

                if i[1]-i[2]<30:
                    draw.text((i[0]-i[2], i[1]-i[2]),match_name.split("_")[0],draw_color,font=font)
                else:
                    draw.text((i[0] - i[2], i[1] - i[2] - 30), match_name.split("_")[0], draw_color, font=font)
            else:
                print("===============NONE======================")

            draw.arc((i[0] - i[2], i[1] - i[2],i[0] + i[2], i[1] + i[2],),0,360,-1)

        image = np.asarray(pil_image)

        cv2.imwrite(os.path.join(test_image_out_dir,img_name),image)
        cv2.imshow("image",image)
        cv2.waitKey(10)
    print("detect_count:",detect_count)
