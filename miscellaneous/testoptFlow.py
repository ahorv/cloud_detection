'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
'''

import numpy as np
import cv2
import glob
from os import path
from os.path import join
from matplotlib import pyplot as plt

rpicam_path = r'C:\Users\tahorvat\PycharmProjects\Segmentation\rpiCam'  # @ Lab

def readImages():
    list_names = []

    for file in sorted(glob.glob(path.join(rpicam_path,'*.jpg'))):
          list_names.append(file)

    return list_names

def main():
    try:
        fig = plt.figure(1)
        ax = fig.add_subplot(1, 1, 1)
        plt.title('Optical Flow')

        # Start the webcam
        list_names = readImages()

        # Take the first frame and convert it to gray
        frame1 = cv2.imread(list_names[0], 1)
        gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        # Get the Shi Tomasi corners to use them as initial reference points
        corners = cv2.goodFeaturesToTrack(gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        #cornerColors = np.random.randint(0, 255, (corners.shape[0], 3))
        color = (0, 255, 0)

        # Create a mask image for drawing purposes
        mask = np.zeros_like(frame1)

        # Define the parameters for Lucas Kanade optical flow
        lkParameters = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 0.03))

        next_img = ax.imshow(frame1)
        fig.show()
        counter = 1

        # Play until the user decides to stop
        while counter < len(list_names):
            # Save the previous frame data
            previousGray = gray
            previousCorners = corners.reshape(-1, 1, 2)

            # Get the next frame
            frame1 = cv2.imread(list_names[counter],1)

            # Convert the frame to gray scale
            gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

            # Calculate optical flow
            corners, st, err = cv2.calcOpticalFlowPyrLK(previousGray, gray, previousCorners, None, **lkParameters)

            # Select only the good corners
            corners = corners[st == 1]
            previousCorners = previousCorners[st == 1]
            #cornerColors[st == 1]

            # Check that there are still some corners left
            if corners.shape[0] == 0:
                print('Stopping. There are no corners left to track')
                break

            # Draw the corner tracks
            for i in range(corners.shape[0]):
                x, y = corners[i]
                xPrev, yPrev = previousCorners[i]
                #color = cornerColors[i].tolist()
                frame1 = cv2.circle(frame1, (x, y), 5, color, -1)
                mask = cv2.line(mask, (x, y), (xPrev, yPrev), color, 2)
            frame1 = cv2.add(frame1, mask)

            # Display the resulting frame
            #cv2.imshow('optical flow', frame1)
            #k = cv2.waitKey(30) & 0xff

            # Exit if the user press ESC
            #if k == 27:
            #    break

            next_img.set_data(frame1)

            fig.canvas.draw()
            plt.pause(.01)

            previousGray = gray.copy()
            counter += 1


        # When everything is done, release the capture and close all windows
        cv2.destroyAllWindows()

    except Exception as e:
        print('MAIN: Error in main: ' + str(e))

if __name__ == '__main__':
    main()
