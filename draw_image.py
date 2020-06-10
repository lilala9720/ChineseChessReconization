import cv2
import os
line_size=80

draw_image = None
def draw_mark_postion():
    global  draw_image
    draw_image = base_image.copy()
    for index,p in enumerate(point_list):
        cv2.circle(draw_image, (p[0], p[1]), int(line_size/2), (255, 0, 0), thickness=-1)
 
    cv2.imshow("show", draw_image)

def on_lower_layer_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    """
    Event Listener
    """
    global position_info_dict
    if event == cv2.EVENT_LBUTTONDOWN:
        point_list.append([x,y])
        draw_mark_postion()


cv2.namedWindow("show",0)
cv2.setMouseCallback("show", on_lower_layer_EVENT_LBUTTONDOWN)

image_dir = 'draw_image'

for i in (os.listdir(image_dir)):
    image_path = os.path.join(image_dir,i)

    base_image = cv2.imread(image_path)

    cv2.imshow("show",base_image)

    point_list = []

    while True: # keep listenign for events
        key = cv2.waitKey(0)
        if key ==13: #（enter） confirm quitting initial interface
            if len(point_list)==4:
                break
        if key ==ord("q"): # undo the action of selecting current point
            
            if len(point_list)>1:  # If the number of selected points is greater than one, then every time we undo the action, the drawing starts at the position of the second last point
                point_list = point_list[:-1]
            else: # If the number of selected points is less than 1, we can set the point_list be empty
                point_list =[]
            draw_mark_postion()
    cv2.imwrite(image_dir+"/"+i,draw_image)
  
