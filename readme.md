A solution to predicting the coordinates of a soccer ball given a video / frame of a FIFA match.

### Quickstart
```bash
git clone https://github.com/mihailgavrilita/ball-detection.git
cd ball-detection
pip install -r requirements.txt
python solution.py your_video.mp4 --visualize
```
And now in a bit more detail :)
Before beginning, ensure that you have installed `git`, `python` and `pip`.
Running in a virtual environment is recommended.
Choose a folder you want to download the project and run:
```bash
git clone https://github.com/mihailgavrilita/ball-detection.git
```
After download, open the project directory:
```bash
cd ball-detection
ls
```
You should see the following project structure:
```
ball-detection
-- classifiers
-- training
-- .gitignore
-- accuracy.py
-- readme.md
-- requirements.txt
-- solution.py
```
Now you can install the dependencies:
```bash
pip install -r requirements.txt
```
After that's done, you should be able to run the `solution.py` file.
Don't forget to save the video in the same folder as the solution script:
```
python solution.py your_video.mp4
```
If you want to see the video while the code is running, set the `--visualize` flag:
```
python solution.py your_video.mp4 --visualize
```
After running, the script will generate a `.csv` file with results in the same folder as the solution script.

### Assumptions
Several assumptions were made: 
- There is only 1 ball on the field;
- The ball is a small, white, round object.

### Approaches
Two approaches were tried:
1. Cascade Classifier approach;
2. Hough Circle Transform approach.

The Cascade Classifier approach seemed promising because it already had proven effective in detecting faces, humans, cars etc.
The only problem remaining was to train your own ball-detecting classifier.
If having enough training data and time, it can detect any type of objects, not only balls.

The Hough Circle Transform approach (which can be found implemented in `solution.py`) seemed promising because it did not require any training.
It also yielded better accuracy at a slight performance cost.
However, it is less flexible than the other approach, being able to only detect circles.

### Tuning and Accuracy

For the Cascade Classifier approach, tuning was performed by training with different samples created by the `opencv_createsamples` app or with samples generated from cropped frames.
After training, tuning would happen by adjusting parameters of the `CascadeClassifier.detectMultiScale()` method.

For the Hough Circle Transform approach, tuning was performed chiefly by adjusting the parameters of the `HoughCircles` class.

The accuracy of the models was tested with `accuracy.py`.
When using the `cascade_900_1_rand_cropped_neg` classifier, while it detected more frames with a ball in them, the hits (distance between detected ball coordinates and true ball coordinates < 5 pixels) consisted only 14%.
In comparison, `HoughCircles` detected fewer frames with a ball, but the percentage of frames with hits grew to 31%.

### Future
For the Cascade Classifier approach:
- Clean-up negative imageset -- some frames repeat (such as the splash screen at the beginning of the video);
- Clean-up positive imageset -- some coordinates are not pointing directly at ball;
- Continue tuning the model.

For the Hough Circle Transform approach:
- Dots on the top-down view may lead to misses -- assume ball is the circle closest to center of screen;
- Continue tuning the model.

### References
1. [Building OpenCV from source](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html);
2. [Docs on how to train your own .xml haar-cascade file](https://docs.opencv.org/4.x/dc/d88/tutorial_traincascade.html);
3. [An article on how to train your own .xml haar-cascade file](https://pythonprogramming.net/haar-cascade-object-detection-python-opencv-tutorial);
4. [Docs on how to detect circles](https://docs.opencv.org/3.4/d4/d70/tutorial_hough_circle.html).
