# import os
# import json
# import sys

# def match_files(image_dir, mask_dir):
#     image_files = os.listdir(image_dir)
#     mask_files = os.listdir(mask_dir)
#     count = 0
#     file_mapping = {}
#     for image_file in image_files:
#         image_name, image_ext = os.path.splitext(image_file)
#         for mask_file in mask_files:
#             mask_name, mask_ext = os.path.splitext(mask_file)
#             if image_name == mask_name.split('.')[0]:
#                 image_path = os.path.join(image_dir, image_file).replace("\\", "/")
#                 mask_path = os.path.join(mask_dir, mask_file).replace("\\", "/")
#                 file_mapping[mask_path] = image_path
#                 count+=1
#     return file_mapping,count



# def save_json(data, output_file):
#     with open(output_file, 'w') as f:
#         json.dump(data, f, indent=4)


# if __name__ == "__main__":
#     data = sys.argv[1]
#     image_dir = f"{data}/images"
#     mask_dir = f"{data}/masks"
#     output_file = f"{data}/label2image_test.json"

#     mapping,count = match_files(image_dir, mask_dir)
#     save_json(mapping, output_file)
#     print(f'total nums:{count}')

import os
import json
import sys

def match_files(image_dir, mask_dir):
    image_files = os.listdir(image_dir)
    mask_files = os.listdir(mask_dir)
    count = 0
    file_mapping = {}
    unmatched_images = set(image_files)  # 用于跟踪未被匹配的图像文件

    for image_file in image_files:
        image_name, image_ext = os.path.splitext(image_file)
        for mask_file in mask_files:
            mask_name, mask_ext = os.path.splitext(mask_file)
            if image_name == mask_name.split('.')[0]:
                image_path = os.path.join(image_dir, image_file).replace("\\", "/")
                mask_path = os.path.join(mask_dir, mask_file).replace("\\", "/")
                file_mapping[mask_path] = image_path
                count += 1
                if image_file in unmatched_images:
                    unmatched_images.remove(image_file)  # 移除已匹配的图像文件

    return file_mapping, count, list(unmatched_images)

def save_json(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    data = sys.argv[1]
    image_dir = f"{data}/images"
    mask_dir = f"{data}/masks"
    output_file = f"{data}/label2image_test.json"
    unmatched_file = f"{data}/unmatched_images.json"

    mapping, count, unmatched_images = match_files(image_dir, mask_dir)
    save_json(mapping, output_file)
    save_json(unmatched_images, unmatched_file)  # 保存未匹配的图像文件
    print(f'Total matched pairs: {count}')
    print(f'Total unmatched images: {len(unmatched_images)}')
