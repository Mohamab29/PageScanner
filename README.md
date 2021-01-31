# PageScanner

The Scanner.py script is an image scanner meaning it is meant to scan images that contain a rectangle paper and we want to scan it like how the app CamScan does it.
<br>
First we take a given image (preferred to have dimensions >(1000,1000) else the resizing needs to be changed),<br>
and then detect its edges using canny edge detector ,<br>after that 
we calculate the location of the four corners of the rectangle using contour detection, 
and after we've found the corners we transform the ROI into a top down view
using OpenCv warpPerspective and finally we threshold the image so it looks like a scanned image

example:

<img src="https://github.com/Mohamab29/PageScanner/blob/main/page.jpg" width="500" height="500">

<hr>

<img src="https://github.com/Mohamab29/PageScanner/blob/main/saved.jpg" width="500" height="500">

<hr>

In order to run the script it need to run in the terminal like this:
<br>
>python Scanner.py "path of an image to be scanned"  "the path you want to save the scanned image at"

Thanks to Dr. Irina Rabaev and Adrian Rosebrock 

Prerequisites:
numpy
cv2
imutils
argparse

PS:
The image needs to contain a rectangle paper.
