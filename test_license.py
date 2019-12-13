from license_model import LicenseNumberDetector
from PIL import Image
import cv2
import os

detector = LicenseNumberDetector()

path = '/home/tupm/Downloads/license_huy_label/license_huy_label'
img_paths = [os.path.join(path, e) for e in os.listdir(path) if '.jpg' in e]
for img_path in img_paths:
    frame = cv2.imread(img_path)
    image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
    bboxs, classes = detector.detect_image(image)
    print(classes)
    for box in bboxs:
        x, y, w, h = box[:4]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('', frame)
    cv2.waitKey(0)
cv2.destroyAllWindows()
