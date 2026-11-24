from ultralytics import YOLO

model = model = YOLO("./best.pt")

metrics = model.val(
    data="./data.yaml",
    split="test",
    imgsz=800,
    name="test"
)

print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)