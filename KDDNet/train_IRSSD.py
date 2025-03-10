import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/KDDNet/KDDNet.yaml')
    model.train(data='dataset/data-IRSSD.yaml',
                pretrained=False,
                cache=False,
                imgsz=640,
                epochs=200,
                batch=4,
                workers=4,
                device='0',
                project='runs/IRSSD',
                name='KDDNet'
                )