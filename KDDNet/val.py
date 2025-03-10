import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('runs/ISDD/KDDNet/weights/best.pt')
    model.val(data='dataset/data-ISDD.yaml',
              split='test',
              imgsz=640,
              batch=4,
              device='0',
              project='runs/val',
              name='KDDNet',
              )