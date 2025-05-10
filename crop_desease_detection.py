# Complete Pipeline for Tree Disease Detection with PDT Dataset

# Cell 1: Install required packages
!pip install ultralytics torch torchvision opencv-python matplotlib
!pip install huggingface_hub

import os
import shutil
import zipfile
from ultralytics import YOLO
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import snapshot_download
from IPython.display import Image, display

# Cell 2: Download the PDT dataset from HuggingFace
print("Downloading PDT dataset from HuggingFace...")

try:
    dataset_path = snapshot_download(
        repo_id='qwer0213/PDT_dataset',
        repo_type='dataset',
        local_dir='/content/PDT_dataset',
        resume_download=True
    )
    print(f"Dataset downloaded to: {dataset_path}")
except Exception as e:
    print(f"Error downloading dataset: {e}")

# Cell 3: Find and extract the zip file
print("\nLooking for zip file in downloaded dataset...")

# Find the zip file
zip_file_path = None
for root, dirs, files in os.walk('/content/PDT_dataset'):
    for file in files:
        if file.endswith('.zip'):
            zip_file_path = os.path.join(root, file)
            print(f"Found zip file: {zip_file_path}")
            break
    if zip_file_path:
        break

if not zip_file_path:
    print("No zip file found in the downloaded dataset!")
else:
    # Extract the zip file
    extract_path = '/content/PDT_dataset_extracted'
    os.makedirs(extract_path, exist_ok=True)
    
    print(f"Extracting {zip_file_path} to {extract_path}")
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction completed!")

# Cell 4: Explore the extracted dataset structure
print("\nExploring dataset structure...")

def explore_dataset_structure(base_path):
    """Explore and find the actual dataset structure"""
    dataset_info = {
        'yolo_txt_path': None,
        'voc_xml_path': None,
        'train_path': None,
        'val_path': None,
        'test_path': None
    }
    
    for root, dirs, files in os.walk(base_path):
        # Look for YOLO_txt directory
        if 'YOLO_txt' in root:
            dataset_info['yolo_txt_path'] = root
            print(f"Found YOLO_txt at: {root}")
            
            # Check for train/val/test
            for split in ['train', 'val', 'test']:
                split_path = os.path.join(root, split)
                if os.path.exists(split_path):
                    dataset_info[f'{split}_path'] = split_path
                    print(f"Found {split} at: {split_path}")
        
        # Look for VOC_xml directory
        if 'VOC_xml' in root:
            dataset_info['voc_xml_path'] = root
            print(f"Found VOC_xml at: {root}")
    
    return dataset_info

dataset_info = explore_dataset_structure('/content/PDT_dataset_extracted')

# Cell 5: Setup YOLO dataset from the PDT dataset
def setup_yolo_dataset(dataset_info, output_dir='/content/PDT_yolo'):
    """Setup YOLO dataset from the extracted PDT dataset"""
    print(f"\nSetting up YOLO dataset to {output_dir}")
    
    # Clean output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)
    
    total_copied = 0
    
    # Process each split
    for split in ['train', 'val', 'test']:
        split_path = dataset_info[f'{split}_path']
        
        if not split_path or not os.path.exists(split_path):
            print(f"Warning: {split} split not found")
            continue
        
        print(f"\nProcessing {split} from: {split_path}")
        
        # Find images and labels directories
        img_dir = os.path.join(split_path, 'images')
        lbl_dir = os.path.join(split_path, 'labels')
        
        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            print(f"Warning: Could not find images or labels for {split}")
            continue
        
        # Copy images and labels
        img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Found {len(img_files)} images in {split}")
        
        for img_file in img_files:
            # Copy image
            src_img = os.path.join(img_dir, img_file)
            dst_img = os.path.join(output_dir, 'images', split, img_file)
            shutil.copy2(src_img, dst_img)
            
            # Copy corresponding label
            base_name = os.path.splitext(img_file)[0]
            txt_file = base_name + '.txt'
            src_txt = os.path.join(lbl_dir, txt_file)
            dst_txt = os.path.join(output_dir, 'labels', split, txt_file)
            
            if os.path.exists(src_txt):
                shutil.copy2(src_txt, dst_txt)
                total_copied += 1
    
    # Create data.yaml
    data_yaml_content = f"""# PDT dataset configuration
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val
test: images/test

# Classes
names:
  0: unhealthy
nc: 1
"""
    
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(data_yaml_content)
    
    print(f"\nDataset setup completed!")
    print(f"Total images copied: {total_copied}")
    
    # Verify the dataset
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(output_dir, 'images', split)
        lbl_dir = os.path.join(output_dir, 'labels', split)
        if os.path.exists(img_dir):
            img_count = len([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
            lbl_count = len([f for f in os.listdir(lbl_dir) if f.endswith('.txt')])
            print(f"{split}: {img_count} images, {lbl_count} labels")
    
    return yaml_path

# Setup the dataset
data_yaml_path = setup_yolo_dataset(dataset_info)

# Cell 6: Train the model
print("\nStarting model training...")

# Use YOLOv8s model
model = YOLO('yolov8s.yaml')

# Train the model
results = model.train(
    data=data_yaml_path,
    epochs=50,  # Adjust based on your needs
    imgsz=640,
    batch=16,  # Adjust based on GPU memory
    name='yolov8s_pdt',
    patience=10,
    save=True,
    device='0' if torch.cuda.is_available() else 'cpu',
    workers=4,
    project='runs/train',
    exist_ok=True,
    pretrained=False,
    optimizer='SGD',
    lr0=0.01,
    momentum=0.9,
    weight_decay=0.001,
    verbose=True,
    plots=True,
)

print("Training completed!")

# Cell 7: Evaluate the model
print("\nEvaluating model performance...")

# Load the best model
best_model_path = 'runs/train/yolov8s_pdt/weights/best.pt'
model = YOLO(best_model_path)

# Validate
metrics = model.val()

print(f"\nValidation Metrics:")
print(f"mAP50: {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")
print(f"Precision: {metrics.box.p.mean():.3f}")
print(f"Recall: {metrics.box.r.mean():.3f}")

# Cell 8: Test the model
print("\nTesting on sample images...")

# Test on validation images
val_img_dir = '/content/PDT_yolo/images/val'
val_images = [f for f in os.listdir(val_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))][:5]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for i, img_name in enumerate(val_images[:6]):
    img_path = os.path.join(val_img_dir, img_name)
    
    # Run inference
    results = model(img_path, conf=0.25)
    
    # Plot results
    img_with_boxes = results[0].plot()
    axes[i].imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
    axes[i].set_title(f'{img_name}')
    axes[i].axis('off')

# Hide empty subplot
if len(val_images) < 6:
    axes[5].axis('off')

plt.tight_layout()
plt.show()

# Cell 9: Create inference function
def detect_tree_disease(image_path, conf_threshold=0.25):
    """Detect unhealthy trees in an image"""
    results = model(image_path, conf=conf_threshold)
    
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                detection = {
                    'confidence': float(box.conf[0]),
                    'bbox': box.xyxy[0].tolist(),
                    'class': 'unhealthy'
                }
                detections.append(detection)
    
    # Visualize
    img_with_boxes = results[0].plot()
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f'Detected {len(detections)} unhealthy tree(s)')
    plt.show()
    
    return detections

# Cell 10: Save the model
print("\nSaving model...")
final_model_path = 'tree_disease_detector.pt'
model.save(final_model_path)
print(f"Model saved to: {final_model_path}")

# Cell 11: Save to Google Drive (optional)
from google.colab import drive

try:
    drive.mount('/content/drive')
    
    save_dir = '/content/drive/MyDrive/tree_disease_detection'
    os.makedirs(save_dir, exist_ok=True)
    
    # Copy files
    shutil.copy(best_model_path, os.path.join(save_dir, 'best_model.pt'))
    shutil.copy(final_model_path, os.path.join(save_dir, 'tree_disease_detector.pt'))
    
    # Copy training results
    results_png = 'runs/train/yolov8s_pdt/results.png'
    if os.path.exists(results_png):
        shutil.copy(results_png, os.path.join(save_dir, 'training_results.png'))
    
    print(f"Results saved to Google Drive: {save_dir}")
except:
    print("Google Drive not mounted. Results saved locally.")

# Cell 12: Summary
print("\n=== Training Complete ===")
print("Model: YOLOv8s")
print("Dataset: PDT (Pests and Diseases Tree)")
print(f"Best Model: {best_model_path}")
print("The model is ready for tree disease detection!")

# Test with your own image
print("\nTo test with your own image:")
print("detections = detect_tree_disease('path/to/your/image.jpg')")