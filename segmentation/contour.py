import numpy as np
from matplotlib import pyplot as plt
import cv2
import math
from copy import deepcopy

global input_path
global output_path
input_path  = r'C:\Users\tahorvat\PycharmProjects\Segmentation\rpiCam\tl_0020_0746_20171027_095526.jpg'  # @ Lab
matCnt_path = r'C:\Users\tahorvat\PycharmProjects\Segmentation\Examples\matlab_contour_tl_0020_0746_20171027_095526.jpg' # @ Lab

#input_path  = r'C:\Hoa_Python_Projects\segmentation\rpiCam\tl_0020_0746_20171027_095526.jpg'   # @ Home
#matCnt_path = r'C:\Hoa_Python_Projects\segmentation\matlab_contour_tl_0020_0746_20171027_095526.jpg' # @ Home

class Segmentation(object):
    def __init__(self):
        # Create image mask
        size = 1944, 2592, 3
        empty_img = np.zeros(size, dtype=np.uint8)
        self.mask = self.cmask([880, 1190], 1117, empty_img)

    def cmask(self, index, radius, array):
        """Generates the mask for a given input image.
        The generated mask is needed to remove occlusions during post-processing steps.

        Args:
            index (numpy array): Array containing the x- and y- co-ordinate of the center of the circular mask.
            radius (float): Radius of the circular mask.
            array (numpy array): Input sky/cloud image for which the mask is generated.

        Returns:
            numpy array: Generated mask image."""

        a, b = index
        is_rgb = len(array.shape)

        if is_rgb == 3:
            ash = array.shape
            nx = ash[0]
            ny = ash[1]
        else:
            nx, ny = array.shape

        s = (nx, ny)
        image_mask = np.zeros(s)
        y, x = np.ogrid[-a:nx - a, -b:ny - b]
        mask = x * x + y * y <= radius * radius
        image_mask[mask] = 1

        return (image_mask)

    def get_masked_img(self,input_image,mask_image):

        red   = input_image[:, :, 0]
        green = input_image[:, :, 1]
        blue  = input_image[:, :, 2]

        r_img = red.astype(float)   * mask_image
        g_img = green.astype(float) * mask_image
        b_img = blue.astype(float) * mask_image

        dimension = (input_image.shape[0], input_image.shape[1], 3)
        output_img = np.zeros(dimension, dtype=np.uint8)

        output_img[..., 0] = r_img[:, :]
        output_img[..., 1] = g_img[:, :]
        output_img[..., 2] = b_img[:, :]

        return output_img

    def maske_OpenCV_Image(self, input_image):

        red   = input_image[:, :, 0]
        green = input_image[:, :, 1]
        blue  = input_image[:, :, 2]

        r_img = red.astype(float)   * self.mask
        g_img = green.astype(float) * self.mask
        b_img = blue.astype(float) * self.mask

        dimension = (input_image.shape[0], input_image.shape[1], 3)
        output_img = np.zeros(dimension, dtype=np.uint8)

        output_img[..., 0] = r_img[:, :]
        output_img[..., 1] = g_img[:, :]
        output_img[..., 2] = b_img[:, :]

        return output_img

    def draw_contour_1(self,image,minArea = 300.0,TrCol = [150, 117],LimitVal = 9):
        'new better version'

        #minArea = 300.0  # minimum size of Area
        #TrCol = [142, 120]
        #LimitVal = 7

        imageYCbCr = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
        imageYCbCr = imageYCbCr.astype(np.double)

        Y = imageYCbCr[..., 0]
        Cb = imageYCbCr[..., 1]
        Cr = imageYCbCr[..., 2]

        index_Img = ((Cb - TrCol[0]) ** 2 + (Cr - TrCol[1]) ** 2) < LimitVal ** 2

        #fig2 = plt.figure(2)
        #plt.title('index Image')
        #plt.imshow(index_Img,cmap ='gray')

        imgray = np.array(index_Img, dtype=np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        closing = cv2.morphologyEx(imgray, cv2.MORPH_CLOSE, kernel)

        #fig3 = plt.figure(3)
        #plt.title('closing')
        #plt.imshow(closing,cmap ='gray')

        erode = cv2.morphologyEx(closing, cv2.MORPH_ERODE, kernel)

        #fig4 = plt.figure(4)
        #plt.title('erode')
        #plt.imshow(erode,cmap ='gray')

        # RETR_EXTERNAL
        # RETR_LIST
        # RETR_TREE

        im2, contours, hierarchy = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #frst = 1
        # cv2.RETR_LIST
        #for first in hierarchy:
        #    for second in first:
        #        print('{}: {}'.format(frst,second))

        #       frst+=1
        #cnt = (contours[0][:])

        cleaned_contours = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if (area > minArea):
                cleaned_contours.append(cnt)

        img_cleand_contours = cv2.drawContours(deepcopy(image), cleaned_contours, -1, (0, 0, 0), 3)


        return img_cleand_contours

    def draw_contour_2(self,image):
        'old version'
        imageYCbCr = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
        imageYCbCr = imageYCbCr.astype(np.double)

        Y = imageYCbCr[..., 0]
        Cb = imageYCbCr[..., 1]
        Cr = imageYCbCr[..., 2]

        TrCol = [140, 120]

        LimitVal = 7

        red = image[..., 0]
        green = image[..., 1]
        blue = image[..., 2]

        index_Img = ((Cb - TrCol[0]) ** 2 + (Cr - TrCol[1]) ** 2) < LimitVal ** 2

        red[index_Img] = 255
        green[index_Img] = 0
        blue[index_Img] = 0

        dimension = (image.shape[0], image.shape[1], 3)
        subtracted_img = np.zeros(dimension, dtype=np.uint8)

        subtracted_img[..., 0] = red[:, :]
        subtracted_img[..., 1] = green[:, :]
        subtracted_img[..., 2] = blue[:, :]

        imgray = cv2.cvtColor(subtracted_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(imgray, (5, 5), 0)

        fig0 = plt.figure(0)
        plt.title('Gray with blur Image')
        plt.imshow(imgray,cmap ='gray')

        ret, thresh = cv2.threshold(imgray, 29, 255, 0)

        fig2 = plt.figure(2)
        plt.title('Thresholded Image')
        plt.imshow(thresh,cmap ='gray')

        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img_with_contour = cv2.drawContours(image, contours, -1, (0, 255, 0), 3)


        return img_with_contour

    def blendImages(self, img1,img2):
        alpha = 0.5  # must be [0,1]
        beta = (1.0 - alpha)

        #print('img1: shape: {}'.format(img1.shape))
        #print('img2: shape: {}'.format(img2.shape))

        superimposed = cv2.addWeighted(img1, alpha, img2, beta, 0.0)

        fig6 = plt.figure(6)
        plt.title('Superimposed')
        plt.imshow(superimposed)
        plt.show()

        return superimposed

def main():
    try:
        img = cv2.imread(input_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        seg = Segmentation()
        masked_img = seg.maske_OpenCV_Image(img)

        img_with_contour = seg.draw_contour_1(masked_img,TrCol = [142, 120],LimitVal = 7)

        matcnt = cv2.imread(matCnt_path)
        matContour = cv2.cvtColor(matcnt,cv2.COLOR_BGR2RGB)
        maked_matContour = seg.maske_OpenCV_Image(matContour)

        seg.blendImages(maked_matContour,img_with_contour)

    except Exception as e:
        print('Error in main: {}'.format(e))


if __name__ == '__main__':
    main()