# Finding Lane Lines on the Road



## Objective

The objectives of this project are the following:

* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

## Process Pipeline Breakdown
### Dependencies
This project rely on the following Python packages

- math
- moviepy
- numpy
- opencv
- request
- scipy

### Pre-process
The input image/frame is converted into grayscale, and then Gaussian blurred to smooth insignificant edges, reducing potential noise in Canny edge detection.

Practical speaking, converting into grayscale, discarding colour infomation is not necessarily a good idea, as this will cause detection failure in some cases e.g. certain lighting condition. But these conditions will be addressed in the advanced lane line detection project, hence ignored here.

```python
gray = grayscale(image)
blur = gaussian_blur(gray,3)
```
### Edge Detection

```python
edge = canny(blur, 50, 150)
```
### Region of Interest
Cropping region of interest will significantly reduce noise, making following process much easier. 
Here, I crop the trapezoid area define in the code below.

**NOTE:** Apply RoI crop after edge detection, otherwise the crop edge will be detected.

```python
roi = np.copy(edge)

# Define a trapezoid region of interest 
# Keep in mind the origin (x=0, y=0) is in the upper left in image processing
left_bottom = [110, 540]
right_bottom = [890, 540]
left_top = [475, 320]
right_top = [505, 320]

# Fit lines (y=Ax+B) to identify the 4 sided region of interest
# np.polyfit() returns the coefficients [A, B] of the fit
fit_left = np.polyfit((left_bottom[0], left_top[0]), (left_bottom[1], left_top[1]), 1)
fit_right = np.polyfit((right_bottom[0], right_top[0]), (right_bottom[1], right_top[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)
fit_top = np.polyfit((left_top[0], right_top[0]), (left_top[1], right_top[1]), 1)

# Find the region inside the lines
XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
region_thresholds = (YY < (XX*fit_left[0] + fit_left[1])) | \
                    (YY < (XX*fit_right[0] + fit_right[1])) | \
                    (YY > (XX*fit_bottom[0] + fit_bottom[1])) | \
                    (YY < (XX*fit_top[0] + fit_top[1]))

# Color pixels red which are inside the region of interest
roi[region_thresholds] = 0
```
### Morphology Operation
Morphology operations include erode, dilate, open and close. 

- Erode
- Dilate
- Open
- Close


```python
morph = cv2.morphologyEx(roi, cv2.MORPH_OPEN, (3,3))
dilt = cv2.dilate(morph, (3,3))
morph = cv2.morphologyEx(dilt, cv2.MORPH_OPEN, (3,3))
dilt = cv2.dilate(morph, (3,3))
morph = cv2.morphologyEx(dilt, cv2.MORPH_OPEN, (3,3))
dilt = cv2.dilate(morph, (3,3))
```
### Hough Line Transform
Hough Transform convert pixels from spatial dimensions into shape parameter voting dimensions, e.g. rho and theta for```rho = x * cos(theta) + y * sin(theta)``` in line detection.

In OpenCV, the method provides some extra parameters such as `min_line_length` and `max_line_gap` to filter the output, returning more relevent result.

```python
rho = 1
theta = np.pi/180
threshold = 1
min_line_len = 21
max_line_gap = 7
line_image = hough_lines(dilt, rho, theta, threshold, min_line_len, max_line_gap)

result = weighted_img(line_image, image, α=0.8, β=1., γ=0.)
```
### Modifications Made to Draw_Line()
For lane lines, merely applying Hough transform returns lines in segments rather than one continuous line for each side of the lane. Multiple methods can be used to addressed the problem, to calculate average slope for th eleft side and the right side repectively, or find the slope for each side via linear regression. Here, I implemented the latter approach. The SciPy library provides a simple function to fit the line with minimum MSE respect to the given points. The method appears to work fine in this case.

```python
def draw_lines(img, lines, color=[255, 0, 0], thickness=8):
#     print(lines)
    xl = []
    yl = []
    xr = []
    yr = []
    
    lines_left = [line for line in lines for x1,y1,x2,y2 in line if ((y2-y1)/(x2-x1)) < 0]
    lines_right = [line for line in lines for x1,y1,x2,y2 in line if ((y2-y1)/(x2-x1)) > 0]
    
    for line in lines_left:
        for x1,y1,x2,y2 in line:
            xl.append(x1)
            xl.append(x2)
            yl.append(y1)
            yl.append(y2)
            
    for line in lines_right:
        for x1,y1,x2,y2 in line:
            xr.append(x1)
            xr.append(x2)
            yr.append(y1)
            yr.append(y2)
    try:
        slp_left, intercept_left, r_value, p_value, std_err = stats.linregress(xl, yl)
        slp_right, intercept_right, r_value, p_value, std_err = stats.linregress(xr, yr)
        
        xlmin = math.floor(min(xl))
        xrmin = math.floor(min(xr))
        xlmax = math.floor(max(xl))
        xrmax = math.floor(max(xr))
        
        ylmin = math.floor(intercept_left + slp_left*xlmin)
        ylmax = math.floor(intercept_left + slp_left*xlmax)
        yrmin = math.floor(intercept_right + slp_right*xrmin)
        yrmax = math.floor(intercept_right + slp_right*xrmax)
        
        cv2.line(img, (xlmin, ylmin), (xlmax, ylmax), color, thickness)
        
        cv2.line(img, (xrmin, yrmin), (xrmax, yrmax), color, thickness)
    except ValueError:
        pass
```

## Reflection

### 1. Potential Shortcomings with Current Pipeline

The pipeline is quite straightforward, but even with optimised parameter settings, it only works fine under very limited circumstances. Fundamentally, it can neither detect slight curves, nor line marks of certain colour or under certain lighting such as shadows and strong direct sunlight. While occasionally, it might even fail due to bad road condition, not to mention detection when interacting with other vehicles or pedestrians. But nevertheless, the project serves well as a warm up for the course.

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Possible Improvements
With the shortcomings said above, the improvements are straightforward as well.

- Detect in colour channels such as RGB and HSV, and then merge the result
- Another possible way to utilize colour infomation is to convert grayscale images with designed weights
- Curvature detection involves many other techniques such as perspective transform and et c.
- Theoretically, deep learning can nicely solve this problem end-to-end with sufficient and well labeled data

Some of these improvements will be implemented in the following Advanced Lane Detection project.


