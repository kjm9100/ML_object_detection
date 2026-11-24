from ultralytics import YOLO

model = YOLO("yolo26n.pt")

model.train(
    data="./data.yaml",
    epochs=120,
    imgsz=800,           
    batch=8,             
    optimizer="AdamW",
    lr0=0.0012,
    lrf=0.01,
    freeze=0,
    mosaic=0.4,
    close_mosaic=20,
    mixup=0.0,
    copy_paste=0.0,
    degrees=20.0,        
    scale=0.3,           
    fliplr=0.5,
    flipud=0.3,          
    hsv_s=0.5,           
    hsv_v=0.4,           
    name="trained",
)

