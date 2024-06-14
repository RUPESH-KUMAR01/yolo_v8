from matplotlib import pyplot as plt
import numpy as np
from cfg import DEFAULT_CFG_DICT
from data.datset import YOLODataset, check_det_dataset
def li():
    dataset=YOLODataset(check_det_dataset("VOC.yaml")["train"],data=DEFAULT_CFG_DICT)
    plt.imshow(np.transpose(dataset[150]["img"],(1,2,0)))
    plt.axis('off')  # Hide axes
    plt.show()
    print(dataset[150])

li()
