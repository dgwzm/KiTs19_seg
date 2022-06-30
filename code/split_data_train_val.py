import os

data_path=r"D:\torch_keras_code\kits_data_label\data_jpg_0"
label_path=r"D:\torch_keras_code\kits_data_label\label_jpg"

data_file_list=os.listdir(data_path)
data_file_lists=sorted(data_file_list)
data_txt_f=open("../un-et/logs/data_jpg_0.txt", 'w')
data_file_id={}

label_file_list=os.listdir(label_path)
label_file_lists=sorted(label_file_list)
label_txt_f=open("../un-et/logs/label_jpg_0.txt", 'w')
label_file_id={}

for i in data_file_lists:
    data_file_id[i[-5:]]=[]
    d_dir=os.path.join(data_path,i)
    jpg_lists=sorted(os.listdir(d_dir))
    for j in jpg_lists:
        data_file_id[i[-5:]].append(j)
        #w=os.path.join(d_dir,j)
        w=i+'/'+j
        data_txt_f.writelines(w+"\n")
data_txt_f.close()

for i in label_file_lists:
    f_id=i[-5:]
    d_dir=os.path.join(label_path,i)
    jpg_lists=sorted(os.listdir(d_dir))
    for j in jpg_lists:
        if j in data_file_id[f_id]:
            w=os.path.join(d_dir,j)
            w=i+'/'+j
            label_txt_f.writelines(w+"\n")
        else:
            raise "Error"
label_txt_f.close()
