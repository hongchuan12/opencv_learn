from ultralytics import YOLO


model = YOLO('yolov8n.pt') 

if __name__ == '__main__':

    model.train(
        data='dataset/data.yaml', 
        epochs=100,
        imgsz=640,
        batch=16,                 
        device='0',               
        workers=4,                
        amp=True                  
    )
