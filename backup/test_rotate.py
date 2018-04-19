import cv2
import numpy as np
from pathlib import Path
import imutils

# bbox = [43, 56, 142, 230]
# xmin, ymin, bb_width, bb_height = bbox

# img_raw = cv2.imread('./test1.jpg')
# rows, cols = img_raw.shape[:2]

# rotated_90 = imutils.rotate_bound(img_raw, 270)
# cv2.imwrite('rotated_90_v1.jpg', rotated_90)
# rot_minx, rot_miny, rot_width, rot_height = bbox[1], cols-bbox[0]-bbox[2], bbox[3], bbox[2]
# rotated_90_rect = cv2.rectangle(rotated_90, (rot_minx, rot_miny), (rot_minx+rot_width, rot_miny+rot_height), (255, 0, 0), 2)
# cv2.imwrite('rotated_90_rect_v1.jpg', rotated_90_rect)

{'width': 512, 'id': 12525529972713331968, 'file_name': 'Images/skirt_length_labels/91d0d1df107906ea5f61a2658ae7e287.jpg', 'height': 682}
{'width': 512, 'file_name': 'Images/sleeve_length_labels/2a4300eea46c7a0108fdda76138222a0.jpg', 'height': 512, 'id': 1406531919743425796}
{'width': 512, 'file_name': 'Images/sleeve_length_labels/97063b417395920e212d0247ddde98cb.jpg', 'height': 512, 'id': 4373870993551142828}

image_names = ['Images/skirt_length_labels/91d0d1df107906ea5f61a2658ae7e287.jpg']
img_name = image_names[0]

dataset_path = '/data/david/fai_attr/raw_data/test_v1'
outout_path = '/data/david/fai_attr/transfered_data/test_v1'

img_name = 'Images/sleeve_length_labels/97063b417395920e212d0247ddde98cb.jpg'
img_raw = cv2.imread(Path(dataset_path, img_name).as_posix())
img_raw=cv2.resize(img_raw, (512, 512))
cv2.imwrite(Path(outout_path, img_name).as_posix(), img_raw)


