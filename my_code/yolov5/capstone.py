import torch
import wandb
# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp6/weights/best.pt',force_reload=True)  # or yolov5n - yolov5x6, custom

# Images
img = '/Users/kamegbor/Documents/VirtualEnvs/apps/capstone/my_code/data_object_image_2 2/training/image_2/003732.png'  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.show()  # or .show(), .save(), .crop(), .pandas(), etc.
# wandb.init(id='1wpxc2gi', resume="must")
# wandb sync