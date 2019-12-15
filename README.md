# Computer Vision Coursework
> by vzbf32


## Requirements 
### Directory Structure
Make sure you have the following directory structure:
```
project-root/
    README.md
    code/
        app.py
        disparity_engine.py
        helpers.py
        preprocessor.py
        yolo.py
        (coco.names)
        (yolov3.cfg)
        (yolov3.weights)
    (TTBB-durham-02-10-17-sub10/)
        ...
```

Items surrounded in parentheses (e.g. `(coco.names)`) above are **not included** as part of
the source code for this coursework and should be added manually.

You can modify the location of the dataset by updating the `master_path_to_dataset` variable
at the top of `app.py`. 


### Versions
- Python 3.7.x
- OpenCV 4.1.x


## Quick Start
### On macOS
```
cd code/
pip3 install opencv-python
python3 app.py
```

### On Linux
```
cd code/
opencv4.1.1.init
python3 app.py
```
