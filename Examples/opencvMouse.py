import cv2
from win32api import GetSystemMetrics

#the [x, y] for each right-click event will be stored here
right_clicks = list()

#this function will be called whenever the mouse is right-clicked
def mouse_callback(event, x, y, flags, params):

    #right-click event value is 2
    if event == 1:
        global right_clicks

        #store the coordinates of the right-click event
        right_clicks.append([x, y])

        #this just verifies that the mouse data is being collected
        #you probably want to remove this later
        print(right_clicks)

path_image = r'C:\Hoa_Python_Projects\Segmentation\rpiCam\tl_0020_0746_20171027_095526.jpg'
img = cv2.imread(path_image,0)

width = GetSystemMetrics(0)
height = GetSystemMetrics(1)
scale_width = 640 / img.shape[1]
scale_height = 480 / img.shape[0]

scale = min(scale_width, scale_height)
window_width = int(img.shape[1] * scale)
window_height = int(img.shape[0] * scale)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', window_width, window_height)

#set mouse callback function for window
cv2.setMouseCallback('image', mouse_callback)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

