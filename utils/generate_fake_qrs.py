import cv2
import os
import albumentations as A
import random

# --- CONFIGURATION ---
INPUT_FOLDER = "../dataset/new"
OUTPUT_FOLDER = "../dataset/new"
FAKES_PER_IMAGE = 1   # 👈 CHANGE THIS (1 or 2 is ideal)
# ---------------------

def get_augmentation_pipelines():
    pipelines = [
        A.Compose([
            A.Perspective(scale=(0.05, 0.1), keep_size=True, p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MedianBlur(blur_limit=3, p=0.5),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            A.RandomBrightnessContrast(p=1.0)
        ]),
        A.Compose([
            A.ImageCompression(quality_range=(20, 40), p=1.0),
            A.GaussNoise(std_range=(0.01, 0.05), per_channel=True, p=1.0),
            A.ToGray(p=1.0)
        ]),
        A.Compose([
            A.Downscale(scale_range=(0.25, 0.5),
                        interpolation_pair={'downscale': cv2.INTER_LINEAR, 'upscale': cv2.INTER_LINEAR},
                        p=1.0),
            A.ImageCompression(quality_range=(10, 30), p=1.0)
        ]),
        A.Compose([
            A.RandomShadow(p=1.0),
            A.Rotate(limit=15, p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0)
        ])
    ]
    return pipelines

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    pipelines = get_augmentation_pipelines()

    for filename in os.listdir(INPUT_FOLDER):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = cv2.imread(os.path.join(INPUT_FOLDER, filename))

            # Randomly choose N pipelines
            selected = random.sample(pipelines, FAKES_PER_IMAGE)

            for idx, pipeline in enumerate(selected):
                augmented = pipeline(image=image)['image']
                save_name = f"{os.path.splitext(filename)[0]}_fake{idx}.jpg"
                cv2.imwrite(os.path.join(OUTPUT_FOLDER, save_name), augmented)

    # Balance report
    orig = len(os.listdir(INPUT_FOLDER))
    fake = len(os.listdir(OUTPUT_FOLDER))
    print("\n=== Class Balance ===")
    print(f"Original images: {orig}")
    print(f"Fake images: {fake}")
    print(f"Ratio (Original:Fake): 1:{fake/orig:.1f}")

if __name__ == "__main__":
    main()
