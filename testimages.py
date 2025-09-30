# test_images.py
import os


def test_image_paths():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    images_dir = os.path.join(base_dir, "images", "extracted")
    outputs_dir = os.path.join(base_dir, "outputs")

    print(f"Project root: {base_dir}")
    print(f"Images dir: {images_dir}")
    print(f"Outputs dir: {outputs_dir}")

    if os.path.exists(images_dir):
        images = os.listdir(images_dir)
        print(f"Found {len(images)} images:")
        for img in images[:5]:  # Show first 5
            print(f"  - {img}")
    else:
        print("❌ Images directory not found!")

    # Test relative path
    rel_path = os.path.relpath(images_dir, outputs_dir)
    print(f"Relative path from outputs to images: {rel_path}")


if __name__ == "__main__":
    test_image_paths()
