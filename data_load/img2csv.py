import pandas as pd
import os
import argparse

"""
main function is to create data csv file which indicates each file's class
and the output's column will be like 'index, id, class_name, label'

python data_load/img2csv.py --img_path data/spc_smokingcalling_20210119/ --classes normal,normal2,normal3,smoking,calling,calling2 --csv data/spc_smokingcalling_20210119.csv
python data_load/img2csv.py --img_path data/spc_smokingcalling_20210203_small_2/ --classes normal,normal2,normal3,smoking,calling,calling2 --csv data/spc_smokingcalling_20210203_2.csv
python data_load/img2csv.py --img_path data/spc_smokingcalling_20210308_addnormal4_3/ --classes normal,normal2,normal3,normal4,smoking,calling,calling2,calling3,calling4 --csv data_inference/spc_smokingcalling_20210308_addnormal4_3.csv
python data_load/img2csv.py --img_path data/goggles_classification/val/ --classes w_goggles,wo_goggles --csv data/goggles_classification_val.csv 
"""

"""---options---"""
parser = argparse.ArgumentParser(description='preprocess Data')
parser.add_argument('--img_path', type=str, required=True, help='enter the image main path')
parser.add_argument('--classes', type=str, required=True, help='enter the class inorder e.g. person, car, dog, cat')
parser.add_argument('--csv', type=str, default='./output.csv', help='where to output csv')
args = parser.parse_args()

def get_data_csv(class_path_list, class_name_list, output_path):

    class_list = []
    for path in class_path_list:
        path = path + '/'
        temp_class_list = [img for img in os.listdir(path)] #get imgs in a list
        temp_class_list.sort()
        class_list.append(temp_class_list) #get each class's imgs


    class_dataframe = []
    for i, path in enumerate(class_list):
        class_dict = {'id': path, 'class_name': class_name_list[i], 'label':i}
        class_df = pd.DataFrame(data=class_dict)
        class_dataframe.append(class_df)

    #concate
    pd.set_option('display.max_rows', None)
    train_csv = pd.concat(class_dataframe, axis=0)

    train_csv.to_csv(output_path)

if __name__ == "__main__":
    class_list = args.classes.split(',')
    img_path_list = [args.img_path + class_name for class_name in class_list] #each class img path
    get_data_csv(img_path_list, class_list, args.csv)