import numpy as np
from matplotlib import pyplot as plt
import cv2

global input_path
global output_path
#input_path  = r'C:\Users\tahorvat\PycharmProjects\Segmentation\rpiCam\tl_0020_0723_20171027_095134.jpg'  # @ Lab
input_path  = r'C:\PycharmProjects\Segmentation\rpiCam\tl_0020_0723_20171027_095134.jpg'   # @ Home


class Segmentation(object):
    isImageIsValid = False

    def backgroudsubstractor(self, image):

        try:
            imageYCbCr = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
            imageYCbCr = imageYCbCr.astype(np.double)

            Y  = imageYCbCr[..., 0]
            Cb = imageYCbCr[..., 1]
            Cr = imageYCbCr[..., 2]

            # TrCol = [134, 125]
            TrCol = [130, 125]

            LimitVal = 10

            red   = image[..., 0]
            green = image[..., 1]
            blue  = image[..., 2]

            index_Img = ((Cb -TrCol[0])**2 + (Cr - TrCol[1])**2) < LimitVal**2

            red[index_Img]   = 255
            green[index_Img] = 0
            blue[index_Img]  = 0

            dimension = (image.shape[0], image.shape[1], 3)
            subtracted_img = np.zeros(dimension, dtype=np.uint8)

            subtracted_img[..., 0] = red[:,:]
            subtracted_img[..., 1] = green[:,:]
            subtracted_img[..., 2] = blue[:,:]

            return subtracted_img

        except Exception as e:
            print('Error in backgroundsubstraction: {}'.format(e))

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



def main():
    try:
        img = cv2.imread(input_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        seg = Segmentation()
        img_seg = seg.backgroudsubstractor(img)

        image_mask = seg.cmask([880, 1190], 1117, img) #972, 1296], 1200,
        masked_img = seg.get_masked_img(img,image_mask)

        fig1 = plt.figure(1)
        plt.title('Segmented Sky')
        plt.imshow(img_seg)
        #plt.imshow(img)

        #fig2 = plt.figure(2)
        #plt.title('cmask')
        #plt.imshow(image_mask,cmap ='gray')

        fig3 = plt.figure(3)
        plt.title('Masked image')
        plt.imshow(masked_img)

        plt.show()

    except Exception as e:
        print('Error in main: {}'.format(e))


if __name__ == '__main__':
    main()