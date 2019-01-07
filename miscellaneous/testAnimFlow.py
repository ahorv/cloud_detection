import cv2
import glob, os
from os.path import join
import numpy as np
from matplotlib import pyplot as plt
#from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition, mark_inset)
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredDrawingArea


rpicam_path = r'C:\Users\tahorvat\PycharmProjects\Segmentation\rpiCam'  # @ Lab
#rpicam_path = r'C:\Hoa_Python_Projects\segmentation\rpiCam'  # @ home


global hue
global saturation
global value

hue = 0
saturation = 0
value = 0

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1)
    fx, fy = flow[y.astype(np.int64), x.astype(np.int64)].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    #vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)   # original image
    #vis = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cv2.polylines(img, lines, 0, (0, 255, 0))   # img -vis
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(img, (x1, y1), 1, (0, 255, 0), -1)
    return img


def visualize_flow(flow):
    magnitude, angle = cv2.cartToPolar(flow[... , 0], flow[... , 1], angleInDegrees=True)

    # dimension for the HSV image
    dimension = (flow.shape[0], flow.shape[1], 3)

    # create HSV image with the same size as frame 1
    # initialize all pixels with zero
    hsv = np.zeros(dimension, dtype=np.uint8)

    hue        = angle / 2
    saturation = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    value      = 255

    hsv[... , 0] = hue        # range [0, 179] # angle of the optical flow-vector (Farbton)
    hsv[... , 1] = saturation # range [0, 255] # normalized magnitude of the flow-vector (Sättigung)
    hsv[... , 2] = value      # range [0, 255] # Platzhalter mit konstantem Wert 255

    '''
    Zum besseren Verständniss fuer die Farbcodierung siehe:
    https://www.rapidtables.com/web/color/RGB_Color.html
    http://blog.helmutkarger.de/raspberry-video-camera-teil-17-exkurs-wie-computer-farben-sehen/
    '''

    # convert HSV image back to RGB space
    hsv_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return hsv_flow

def blend_images(src , overlay , pos=(0,0),scale = 1):
    '''
    c_s = src.shape
    print('Shape src: width: {} | height {} | channels {}'.format(c_s[0],c_s[1],c_s[2]))
    o_s = overlay.shape
    print('Shape overlay: width: {} | height {} | channels {}'.format(o_s[0],o_s[1],o_s[2]))
    '''

    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape  # Size of foreground
    rows, cols, _ = src.shape  # Size of background Image
    y, x = pos[0], pos[1]  # Position of foreground/overlay image

    # loop over all pixels and apply the blending equation
    for i in range(h):
        for j in range(w):
            if x + i >= rows or y + j >= cols:
                continue
            alpha = float(overlay[i][j][2] / 255.0)  # read the alpha channel
            src[x + i][y + j] = alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]

    return src

def readImages():
    list_names = []

    for file in sorted(glob.glob(join(rpicam_path,'*.jpg'))):
          list_names.append(file)

    return list_names

def main():
    try:
        fig = plt.figure(1)
        ax = fig.add_subplot(1, 1, 1)
        plt.title('Optical Flow')

        # inset Axes
        inset = fig.add_axes([0.75,0.75,0.3,0.2], frameon=False)   #[posx,posy, width,hight]
        inset.axes.get_xaxis().set_visible(False)
        inset.axes.get_yaxis().set_visible(False)

        list_names = readImages()

        frame1 = cv2.imread(list_names[0], 1)
        frame1 = cv2.resize(frame1, None, fx=0.25, fy=0.25)
        prev   = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        next_img  = ax.imshow(frame1)
        next_flow = inset.imshow(frame1)    # wenn inset verwendet wird

        fig.show()

        counter = 1

        while counter < len(list_names):

            print('\r{} Frame '.format(counter), end="")

            frame2 = cv2.imread(list_names[counter],1)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            frame2 = cv2.resize(frame2, None, fx=0.25, fy=0.25)
            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

          # flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0) # orginal
            flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 5, 15, 3, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

            #mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])   # Obtain the flow magnitude and direction angle

            #Plot graph i inset
            #inset.cla()
            #inset.plot(mag[-1], color='green')


            # ohne blending
            hsv_flow = visualize_flow(flow)
            next_flow.set_data(hsv_flow)

            imgWithFlow_gray = draw_flow(frame2, flow)  # frame2 -> next
            next_img.set_data(imgWithFlow_gray)

            fig.canvas.draw()
            plt.pause(.01)

            # Make next frame previous frame
            prev = next.copy()
            counter += 1

            if counter >= len(list_names):
                counter = 1
                prev = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)


        print('done !')

    except Exception as e:
        print('MAIN: Error in main: ' + str(e))


if __name__ == '__main__':
    main()




