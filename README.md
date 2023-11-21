# python-opencv
to run the regiodetection.py always use these commands # If you want to save results
python yolov8_region_counter.py --source "path/to/video.mp4" --save-img --view-img

# If you want to run model on CPU
python yolov8_region_counter.py --source "path/to/video.mp4" --save-img --view-img --device cpu

# If you want to change model file
python yolov8_region_counter.py --source "path/to/video.mp4" --save-img --weights "path/to/model.pt"

# If you dont want to save results
python yolov8_region_counter.py --source "path/to/video.mp4" --view-img
#for example
python regiondetection.py --source "/Users/pascal/Desktop/seppevideo.mp4" --save-img --view-img
