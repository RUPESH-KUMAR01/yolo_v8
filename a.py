
from matplotlib import patches, pyplot as plt
import torch
from cfg import DEFAULT_CFG_DICT
from data.build import build_dataloader
from data.dataset import YOLODataset, check_det_dataset
from model.train import DetectionTrainer
import time
# dataset=YOLODataset(
#     img_path=check_det_dataset("VOC.yaml")["train"],
#     data=DEFAULT_CFG_DICT
# )
# loader=build_dataloader(dataset=dataset,batch=16,workers=0)


# output=dataset[1000]
# print(output)
# # Convert image tensor to numpy array
# img_np = output['img'].numpy().transpose(1, 2, 0)

# # Extract bounding box and rescale to image dimensions
# bboxes = output['bboxes'].numpy()
# img_height, img_width = output['resized_shape']

# # Rescale bounding box coordinates
# bboxes[:, 0] *= img_width  # x_center
# bboxes[:, 1] *= img_height # y_center
# bboxes[:, 2] *= img_width  # width
# bboxes[:, 3] *= img_height # height

# # Convert bounding box from center format (x_center, y_center, width, height) to (x_min, y_min, width, height)
# bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2  # x_min
# bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2  # y_min

# # Plot the image
# fig, ax = plt.subplots(1)
# ax.imshow(img_np)

# # Draw bounding boxes
# for bbox in bboxes:
#     rect = patches.Rectangle(
#         (bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', facecolor='none'
#     )
#     ax.add_patch(rect)

# plt.axis('off')
# plt.show()
"""
training test
"""
trainer=DetectionTrainer()
start_time=time.time()
results=trainer.predict(source=r"C:\Users\thata\intern\code\pre-built-models\modified\classroom.mp4",stream=False)
print(time.time()-start_time)
"""
dataset test
"""
# dataset=YOLODataset(check_det_dataset("VOC.yaml")["train"],data=check_det_dataset("VOC.yaml"))

# print(dataset[0])

