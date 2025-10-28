import os
import json
import sys

def match_files(image_dir, mask_dir):
    image_files = os.listdir(image_dir)
    mask_files = os.listdir(mask_dir)
    count = 0
    file_mapping = {}
    for image_file in image_files:
        image_name, image_ext = os.path.splitext(image_file)
        for mask_file in mask_files:
            mask_name, mask_ext = os.path.splitext(mask_file)
            if image_name == mask_name:
 
                if 'tif' not in image_ext and 'tif' in mask_ext: 
                    continue
                if 'tif' in image_ext and 'tif' not in mask_ext: 
                    continue
                else:
                    image_path = os.path.join(image_dir, image_file).replace("\\", "/")
                    mask_path = os.path.join(mask_dir, mask_file).replace("\\", "/")
                    if image_path not in file_mapping:
                        file_mapping[image_path] = []
                    file_mapping[image_path].append(mask_path)
                    count+=1
                    

    return file_mapping,count


def save_json(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    file = sys.argv[1]
    image_dir = f"{file}/images"
    mask_dir = f"{file}/masks"
    output_file = f"{file}/image2label_train.json"

    mapping,count = match_files(image_dir, mask_dir)
    save_json(mapping, output_file)
    print(f'total file: {count}')