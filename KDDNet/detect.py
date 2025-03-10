import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('runs/IRSSD/KDDNet/weights/best.pt') # select your model.pt path
    model.predict(source='dataset/IRSSD/images/test',
                  project='runs/detect/IRSSD', 
                  name='KDDNet',
                  save=True,
                  save_conf=False, 
                  show_conf=False,
                  line_width=1
                  )