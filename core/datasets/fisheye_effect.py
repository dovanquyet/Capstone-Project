'''
This file is modified from https://github.com/Gil-Mor/iFish/blob/master/fish.py
'''
import imageio
from PIL import Image
import numpy as np
from math import sqrt
import sys
import argparse
import os
import matplotlib.pyplot as plt


def get_src_x_y(x, y, w, h, distortion_coefficient):
    """
    given x,y in dstImg, find the corresponding x,y in srcImg
    """

    # normalize x and y to be in interval of [-1, 1]
    xnd, ynd = float((2*x - w)/w), float((2*y - h)/h)

    # get xn and yn distance from normalized center
    rd = sqrt(xnd**2 + ynd**2)

    # new normalized pixel coordinates
    if 1 - distortion_coefficient*(rd**2) != 0:
        xdu = xnd / (1 - (distortion_coefficient*(rd**2)))
        ydu = ynd / (1 - (distortion_coefficient*(rd**2)))
    else:
        xdu = xnd
        ydu = ynd

    # convert the normalized distorted xdn and ydn back to image pixels
    xu, yu = int(((xdu + 1)*w)/2), int(((ydu + 1)*h)/2)
    return xu, yu


def fish(img, distortion_coefficient, points=[]):
    """
    :type img: numpy.ndarray
    :param distortion_coefficient: The amount of distortion to apply.
    :return: numpy.ndarray - the image with applied effect.
    """

    ## If input image is only BW or RGB convert it to RGBA
    ## So that output 'frame' can be transparent. Don't need that at the moment
    w, h = img.shape[0], img.shape[1]
    if len(img.shape) == 2:
        ## Duplicate the one BW channel twice to create Black and White
        ## RGB image (For each pixel, the 3 channels have the same value)
        bw_channel = np.copy(img)
        img = np.dstack((img, bw_channel))
        img = np.dstack((img, bw_channel))
    # if len(img.shape) == 3 and img.shape[2] == 3:
    #     print("RGB to RGBA")
    #     img = np.dstack((img, np.full((w, h), 255)))

    ## prepare array for dst image
    dstimg = np.zeros_like(img, dtype=np.uint8)+255

    ## floats for calcultions
    w, h = float(w), float(h)
    distort_points = []

    ## easier calcultion if we traverse x, y in dst image
    ## note that in this function, x refers vertical coordinate, y refers horizontal, which is inverse of normal 2D space.
    for x in range(len(dstimg)):
        for y in range(len(dstimg[x])):
            xu, yu = get_src_x_y(x,y, w,h, distortion_coefficient)
            # if new pixel is in bounds copy from source pixel to destination pixel
            if 0 <= xu and xu < img.shape[0] and 0 <= yu and yu < img.shape[1]:
                dstimg[x][y] = img[xu][yu]
            for i in range(len(points)):
                p = points[i]
                if abs(xu - p[1]) <=1 and abs(yu - p[0]) <= 1:
                    distort_points.append([x,y])
                    points.pop(i)
                    break

    return dstimg.astype(np.uint8), distort_points


def parse_args(args=sys.argv[1:]):
    """Parse arguments."""

    parser = argparse.ArgumentParser(
        description="Apply fish-eye effect to images.",
        prog='python3 fish.py')

    parser.add_argument("-i", "--image", help="path to image file."
                        " If no input is given, the supplied example 'grid.jpg' will be used.",
                        type=str, default="grid.jpg")

    parser.add_argument("-o", "--outpath", help="file path to write output to."
                        " format: <path>.<format(jpg,png,etc..)>",
                        type=str, default="fish.png")

    parser.add_argument("-d", "--distortion",
                        help="The distoration coefficient. How much the move pixels from/to the center."
                        " Recommended values are between -1 and 1."
                        " The bigger the distortion, the further pixels will be moved outwars from the center (fisheye)."
                        " The Smaller the distortion, the closer pixels will be move inwards toward the center (rectilinear)."
                        " For example, to reverse the fisheye effect with --distoration 0.5,"
                        " You can run with --distortion -0.3."
                        " Note that due to double processing the result will be somewhat distorted.",
                        type=float, default=0.5)

    return parser.parse_args(args)


def random_fisheye_effect(path_to_img='sample.png', square_texture_size=400, center_loc=[200,200], distortion=0.5, for_visualization=False):
    """
    put img into texture, apply fisheye effect on the whole texture, then crop and return distorted img
    center_loc: location (x,y) of img's center w.r.t to texture, where (x,y) is in 2D-linear-space format
    """
    ## read and resize image, and create white texture
    image = Image.open(path_to_img)
    w, h = image.size
    hmax = 300
    if h > hmax:
        wscaled = w*hmax//h
        image = image.resize((wscaled, hmax))
    image = np.array(image)

    shape = image.shape
    texture = np.zeros([square_texture_size, square_texture_size, shape[2]], dtype=np.uint8) + 255

    if for_visualization:
        num_grid = 5
        for i in range(num_grid+1):
            texture[:,(square_texture_size-1)*i//num_grid,:] = 0
            texture[(square_texture_size-1)*i//num_grid,:,:] = 0

    ## where to paste the image
    top_y = center_loc[1]-shape[0]//2
    left_x = center_loc[0]-shape[1]//2
    bottom_y = top_y+shape[0]
    right_x = left_x+shape[1]
    texture[top_y:bottom_y, left_x:right_x,:] = image
    points = [
        [left_x, top_y],
        [left_x, bottom_y],
        [right_x, bottom_y],
        [right_x, top_y]
    ]
    imageio.imwrite(path_to_img[:-4]+'texture.png', texture)

    ## distort the texture
    distort_texture, distort_points = fish(texture, distortion, points)
    print(distort_points)
    imageio.imwrite(path_to_img[:-4]+'texture_distorted.png', distort_texture)
    # plt.imshow(distort_texture)
    # plt.plot([y for x, y in distort_points], [x for x, y in distort_points], 'bo', markersize=3)
    # plt.savefig('sample_distort.png')
    # plt.show()
    
    ## TODO: crop to distorted figure
    ## only applicable for this case
    dleft = distort_points[0][0]
    dright = distort_points[3][0]
    dtop = distort_points[0][1]
    dbottom = distort_points[1][1]
    cropped_dimage = distort_texture[dleft:dright, dtop:dbottom]
    # plt.imshow(cropped_dimage)
    # plt.show()
    
    imageio.imwrite(path_to_img[:-4]+'distorted.png', cropped_dimage)
    

def main():
    args = parse_args()
    try:
        imgobj = imageio.imread(args.image)
    except Exception as e:
        print(e)
        sys.exit(1)
    if os.path.exists(args.outpath):
        ans = input(
            args.outpath + " exists. File will be overridden. Continue? y/n: ")
        if ans.lower() != 'y':
            print("exiting")
            sys.exit(0)
    
    output_img, _ = fish(imgobj, args.distortion)
    imageio.imwrite(args.outpath, output_img, format='png')


if __name__ == '__main__':
    ''' Rationale
    Observation: workers in construction site almost always stand, so their images in fisheye camera are radius-aligned, and in one radius rather than lies on a diameter -> just need to distort our image in such the way
        Rotate image -> distort -> rotate back is just the same as put the image vertically. 
        Putting the image vertically simplify the code =)) -> do that
    '''
    # with the rationale from observation, we set the parameter as follow
    # randomize s (800 < size < 1500),
    #           y (height of img/2 < y < text_size/2- height of img/2), 
    #           d (-1 <= d <= 1)
    import random

    is_random = True

    for i in range(1,2):
        s = 1000 if not is_random else random.randint(900, 1500)
        y = 200  if not is_random else random.randint(150, s//2 - 300)
        d = 0.5  if not is_random else random.randint(30, 100)/100
        print(f's {s}, y {y}, d {d}')
        random_fisheye_effect(path_to_img=f"test/ ({i}).jpg",
                              square_texture_size=s,
                              center_loc=[s//2,y],
                              distortion=d,
                              for_visualization=True)

    # note that, the farther y from the texture center, the more squeezed the original img are <-> if a worker stands nearer the fisheye camera
    # use different y to simulate workers at different location of construction site