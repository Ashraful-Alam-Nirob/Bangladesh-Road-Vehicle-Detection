"""# getting all the class names on which we will apply SMOTE
import os
from tqdm import tqdm
label_dir = "/media/nirob/Local Disk/dlenigma1/BadODD/labels/val"

classes = ['auto_rickshaw',
"bicycle",
"bus",
"car",
"cart_vehicle",
"construction_vehicle",
"motorbike",
"person",
"priority_vehicle",
"three_wheeler",
"train",
"truck",
"wheelchair"]

object_in_image = {
 'auto_rickshaw': set(),
 'bicycle': set(),
 'bus': set(),
 'car': set(),
 'cart_vehicle': set(),
 'construction_vehicle': set(),
 'motorbike': set(),
 'person': set(),
 'priority_vehicle': set(),
 'three_wheeler': set(),
 'train': set(),
 'truck': set(),
 'wheelchair': set()
}


class_count = {}
file_name_dct = {5: [], 10: [], 12: []}

for file_name in tqdm(os.listdir(label_dir)):
    file_path = os.path.join(label_dir, file_name)

    with open(file_path, 'r') as f:
        for line in f:
            class_id = int(line.split()[0])
            object_in_image[classes[class_id]].add(file_name)
            class_count[class_id] = class_count.get(class_id, 0) + 1
            if class_id == 10 or class_id == 12 or class_id == 5:
                file_name_dct[class_id].append(file_name)

print(sorted(class_count.items()))
print(object_in_image['wheelchair'])
print(object_in_image['train'])


{'maowa_expressway1_1947.txt'}
{'sylhet4_18290.txt', 'sylhet4_18231.txt'}


"""
"""
import os
import re

# Define the folder path
folder_path = '/media/nirob/Local Disk/dlenigma1/BadODD/images/aug'

# Define the pattern to match files to delete
# This regex matches strings that end with _number_number.jpg
pattern_to_delete = re.compile(r'.*_[0-9]+_[0-9]+\.jpg$')

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    # Check if the file matches the pattern
    if pattern_to_delete.match(filename):
        # Full path to the file
        file_path = os.path.join(folder_path, filename)

        # Delete the file
        os.remove(file_path)
        print(f'Deleted: {filename}')
    else:
        print(f'Skipped: {filename}')"""

import os
from tqdm import tqdm
import random
import ImageCopyPaste as icp
from PIL import Image
from icecream import ic



img_dir = "/media/nirob/Local Disk/dlenigma1/BadODD/images/train"
label_dir = "/media/nirob/Local Disk/dlenigma1/BadODD/labels/train"
save_label_dir = "/media/nirob/Local Disk/dlenigma1/BadODD/labels/cpypst"
save_img_dir = "/media/nirob/Local Disk/dlenigma1/BadODD/images/cpypst"

img_class = {
    #0:set(),1:set(),2:set(),3:set(),4:set(),5:set(),6:set(),7:set(),8:set(),9:set(),10:set(),11:set(),12:set()
    0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[]
} # contains all the class present images names

img_cutout = {
    0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[]
}

lst = [9,6,3,11,1,2,8,4,5,12,10]
class_on_images={}
names= os.listdir(img_dir)
#print(names)
for name in names:
    flag=[]
    name = name.split(".")[0]
    with open(os.path.join(label_dir,name+".txt"),'r') as f :
        for lines in f:
            line = int(lines.split()[0]) # class id
            if line in flag: continue
            img_class[line].append(name)
            x=list(map(float,lines.split()))
            x[0]=int(x[0])
            img_cutout[line].append(x)
            flag.append(line)

random.shuffle(names)
#print(names[0])
for name in tqdm(names):
    #print(name)
    name = name.split(".")[0]
    flag = []
    label_file_labels=[]
    with open(os.path.join(label_dir,name+".txt"),'r') as f :
        for lines in f:
            line = int(lines.split()[0])
            flag.append(line)
            x = list(map(float, lines.split()))
            x[0] = int(x[0])
            label_file_labels.append(x)
    target_img_path = os.path.join(img_dir,name+".jpg")
    #print(label_file_labels)
    target_img = Image.open(target_img_path)
    #target_img.show()
    #print(label_file_labels)
    #print(source_img.size)
    for i in lst:
        if i not in flag:
            rnd_idx = random.randrange(len(img_cutout[i])) # random index
            #import ipdb; ipdb.set_trace()
            source_img_path = os.path.join(img_dir,img_class[i][rnd_idx]+".jpg")
            source_img = Image.open(source_img_path)
            class_12_bbox= img_cutout[i][rnd_idx]
            #print(class_12_bbox)
            #crop_and_paste(source_img_path, target_img_path, class_12_bbox, label_file_labels)
            source_img , label_file_label ,idx = icp.crop_and_paste(source_img,target_img,class_12_bbox,label_file_labels)

            if idx !=-1:
                label_file_labels[idx]=list(label_file_label)
            else:
                label_file_labels.append(list(label_file_label))

    #print(label_file_labels)

    #icp.draw_bboxes_on_image(source_img, label_file_labels)
    #start saving files
    #source_img.show()
    source_img.save(os.path.join(save_img_dir,name+"_cpypst.jpg"))
    with open(os.path.join(save_label_dir,name+"_cpypst.txt"),'w') as f :
        for line in label_file_labels:
            bbox_str = " ".join(map(str, line))
            f.write(f"{bbox_str}\n")


# label file label thik korte hobe
