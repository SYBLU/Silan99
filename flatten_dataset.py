import os
import shutil

dataset_root = 'dataset'

for label in os.listdir(dataset_root):
    label_path = os.path.join(dataset_root, label)
    if os.path.isdir(label_path):
        for subdir in os.listdir(label_path):
            subdir_path = os.path.join(label_path, subdir)
            if os.path.isdir(subdir_path):
                print(f"📂 Flattening: {subdir_path}")
                for file in os.listdir(subdir_path):
                    src_file = os.path.join(subdir_path, file)
                    dst_file = os.path.join(label_path, file)
                    try:
                        if os.path.isfile(src_file):
                            shutil.move(src_file, dst_file)
                    except Exception as e:
                        print(f"❌ Error moving {src_file}: {e}")
                try:
                    os.rmdir(subdir_path)
                    print(f"✅ Removed: {subdir_path}")
                except OSError:
                    print(f"⚠️ Could not remove (not empty): {subdir_path}")
