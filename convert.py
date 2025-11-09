import os, shutil

def convert_yolo_to_classification(src_dir, dest_dir):
    images_dir = os.path.join(src_dir, "images")
    labels_dir = os.path.join(src_dir, "labels")
    os.makedirs(dest_dir, exist_ok=True)

    for label_file in os.listdir(labels_dir):
        label_path = os.path.join(labels_dir, label_file)
        if not label_file.endswith(".txt"):
            continue
        
        with open(label_path) as f:
            line = f.readline().strip()
            if not line:
                continue
            cls_id = line.split()[0] 

        class_dir = os.path.join(dest_dir, cls_id)
        os.makedirs(class_dir, exist_ok=True)

        img_name = label_file.replace(".txt", ".jpg")
        img_path = os.path.join(images_dir, img_name)
        if os.path.exists(img_path):
            shutil.copy(img_path, os.path.join(class_dir, img_name))

convert_yolo_to_classification("train", "train_classification")
convert_yolo_to_classification("valid", "valid_classification")
convert_yolo_to_classification("test", "test_classification")