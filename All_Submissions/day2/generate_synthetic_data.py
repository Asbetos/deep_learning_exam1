
import os
import numpy as np
import pandas as pd
from tensorflow import keras
from PIL import Image

# ============================================================================
# GENERATE SYNTHETIC IMAGES FOR MINORITY CLASSES
# ============================================================================

# Configuration
LATENT_DIM = 128
NUM_CLASSES = 10
IMAGE_SIZE = (100, 100)

# Class distribution from training data
CLASS_DISTRIBUTION = {
    'class1': 22,      # Minority - needs 1,435 samples
    'class6': 63,      # Minority - needs 1,394 samples
    'class10': 736,    # Minority - needs 721 samples
    'class9': 935,     # Minority - needs 522 samples
    'class4': 1212,    # Minority - needs 245 samples
}

MEDIAN_COUNT = 1457

# Calculate how many synthetic samples needed for each minority class
SYNTHETIC_NEEDED = {
    'class1': 1435,
    'class6': 1394,
    'class10': 721,
    'class9': 522,
    'class4': 245,
}

def generate_and_save_synthetic_images(decoder, output_dir='SyntheticData'):
    """
    Generate synthetic images for all minority classes and save them
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Mapping from class name to class index (0-9)
    class_mapping = {
        'class1': 0, 'class2': 1, 'class3': 2, 'class4': 3, 'class5': 4,
        'class6': 5, 'class7': 6, 'class8': 7, 'class9': 8, 'class10': 9
    }

    synthetic_records = []

    print("=" * 80)
    print("GENERATING SYNTHETIC IMAGES FOR MINORITY CLASSES")
    print("=" * 80)

    for class_name, num_samples in SYNTHETIC_NEEDED.items():
        class_idx = class_mapping[class_name]

        print(f"\nGenerating {num_samples} samples for {class_name} (class_idx={class_idx})...")

        # Sample from latent space
        z_samples = np.random.normal(size=(num_samples, LATENT_DIM))

        # Create one-hot encoded labels
        class_labels = np.zeros((num_samples, NUM_CLASSES))
        class_labels[:, class_idx] = 1

        # Generate images
        generated_images = decoder.predict([z_samples, class_labels], verbose=0)

        # Save each image
        for i, img_array in enumerate(generated_images):
            # Denormalize from [0,1] to [0,255]
            img_array = (img_array * 255).astype(np.uint8)

            # Create PIL Image
            img = Image.fromarray(img_array)

            # Generate filename
            filename = f"synthetic_{class_name}_{i+1:04d}.jpg"
            filepath = os.path.join(output_dir, filename)

            # Save image
            img.save(filepath)

            # Record for Excel
            synthetic_records.append({
                'id': filename,
                'target': class_name,
                'split': 'synthetic',
                'target_class': str([1 if j == class_idx else 0 for j in range(NUM_CLASSES)])
            })

        print(f"  ✓ Generated and saved {num_samples} images")

    # Create DataFrame with synthetic data records
    df_synthetic = pd.DataFrame(synthetic_records)

    # Save to Excel
    synthetic_excel_path = 'synthetic_data_records.xlsx'
    df_synthetic.to_excel(synthetic_excel_path, index=False)
    print(f"\n✓ Synthetic data records saved to: {synthetic_excel_path}")

    print(f"\n✓ Total synthetic images generated: {len(synthetic_records)}")
    print("=" * 80)

    return df_synthetic

def create_balanced_dataset(original_excel='/home/ubuntu/deep_learning_exam1/excel/train_test_cleaned.xlsx',
                           synthetic_excel='synthetic_data_records.xlsx',
                           output_excel='balanced_train_data.xlsx'):
    """
    Combine original training data with synthetic data
    """
    print("\nCreating balanced dataset...")

    # Load original data
    df_original = pd.read_excel(original_excel)
    df_train = df_original[df_original['split'] == 'train'].copy()

    # Load synthetic data
    df_synthetic = pd.read_excel(synthetic_excel)

    # Combine
    df_balanced = pd.concat([df_train, df_synthetic], ignore_index=True)

    # Save
    df_balanced.to_excel(output_excel, index=False)

    print(f"\n✓ Balanced dataset saved to: {output_excel}")
    print(f"  Original training samples: {len(df_train)}")
    print(f"  Synthetic samples: {len(df_synthetic)}")
    print(f"  Total balanced samples: {len(df_balanced)}")

    # Print new class distribution
    print("\nBalanced class distribution:")
    for class_name in sorted(df_balanced['target'].unique()):
        count = len(df_balanced[df_balanced['target'] == class_name])
        print(f"  {class_name}: {count} samples")

    return df_balanced

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Loading trained decoder...")
    decoder = keras.models.load_model('cvae_decoder.keras')
    print("✓ Decoder loaded successfully")

    # Generate synthetic images
    df_synthetic = generate_and_save_synthetic_images(decoder)

    # Create balanced dataset
    df_balanced = create_balanced_dataset()

    print("\n" + "=" * 80)
    print("SYNTHETIC DATA GENERATION COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Use 'balanced_train_data.xlsx' for training your classification MLP")
    print("2. Load images from both 'Data' and 'SyntheticData' folders")
    print("3. Train with the balanced dataset to improve F1 score")
