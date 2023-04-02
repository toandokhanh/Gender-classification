import numpy as np
import csv
import scipy
import scipy.io.wavfile
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn import linear_model
from sklearn.metrics import confusion_matrix, matthews_corrcoef
logistic_cl = linear_model.LogisticRegression()
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import warnings
warnings.warn("deprecated", DeprecationWarning)
warnings.filterwarnings('ignore')
# luu lai thoi gian
from datetime import datetime
from time import gmtime, strftime
import time
## luu lai thoi gian bat dau chay module (s)
start_time = time.time()

today = datetime.today()

import argparse
# Tạo đối tượng ArgumentParser
parser = argparse.ArgumentParser(description="Chương trình xử lý âm thanh", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Thêm đối số cho đường dẫn đến thư mục chứa file âm thanh
parser.add_argument("-p", "--audio_path", default="path/to/audio/folder", help="Đường dẫn đến thư mục chứa file âm thanh") # -p /path/to/audio/folder
parser.add_argument('-e', '--eval_dir', type=str, required=True, help='Đường dẫn đến thư mục cần đánh giá.') # -e path/to/evaluation/dir
parser.add_argument('-r', '--result_file', type=str, required=True, help='Đường dẫn đến file kết quả.') #-r result/fg1.txt
#python3 classification.py -p "path thư mục chứa file âm thanh cần phân loại"  -e "path thư mục chứa âm thanh cần đánh giá kết quả" -r "tên file txt (kết quả đánh giá)"
# vd: python3 classification.py -p wav -e wav/eval -r result/fg.txt
args = parser.parse_args()



def custom_database_import(in_path):
    # Lấy danh sách các tệp âm thanh trong thư mục đầu vào có đuôi .wav
    index_list = [f for f in os.listdir(in_path) if f.endswith('.wav')]
    in_all_audios = []
    in_y = []
    
    # Duyệt qua danh sách các tệp âm thanh
    for filename in index_list:
#       print(filename)
        full_path = os.path.join(in_path, filename)
        rate, data = scipy.io.wavfile.read(full_path, mmap=False)
        in_all_audios.append((rate, data))
        
        # Trích xuất nhãn từ tên tệp và thêm vào danh sách
#       label = filename.split("_")[0] 
        name_parts = filename.split("_")  # tách chuỗi theo dấu gạch dưới
        label = name_parts[0]
#         print('label')
        # print(label)
        in_y.append(label)
    
    # Chuyển danh sách nhãn thành một mảng numpy
    out_y = np.array(in_y)
    
    # Trả về danh sách các tệp âm thanh và mảng numpy chứa các nhãn tương ứng
    return in_all_audios, out_y



def custom_eval_database_import(in_path):
    index_list = os.listdir(in_path)
    in_all_audios = []
    for filename in index_list:
        filename = in_path + f"{filename}"
        in_all_audios.append(scipy.io.wavfile.read(filename, mmap=False))
    return in_all_audios



def custom_csv_print(in_labels, filename):
    list_to_print = []
    for index in range(0, len(in_labels)):
        row_to_print = []
        row_to_print.append(index)
        row_to_print.append(in_labels[index])
        list_to_print.append(row_to_print)
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Id', 'Predicted'])
        for index in range(0, len(list_to_print)):
            writer.writerow(list_to_print[index])
    return



def custom_preprocess(in_all_audios):
    frequency_preprocessed = []
    all_normalized_audios = []
    all_samples_processed = []
    # Normalization
    for i in range(0, len(in_all_audios)):
        single_normalized_audio = in_all_audios[i][1] / np.max(np.abs(in_all_audios[i][1]))
        all_normalized_audios.append(single_normalized_audio)
    # Frequency Domain
    for i in range(0, len(all_normalized_audios)):
        freq = np.abs(np.fft.fft(all_normalized_audios[i]))
        frequency_preprocessed.append(freq[:freq.shape[0]//2])
    
    #update 2/4/2023
    # mỗi mẫu âm thanh sẽ được chia thành in_flag phần bằng nhau
    # Các giá trị trong single_sample_processed được sử dụng để biểu diễn thông tin tần số trung bình của mỗi mẫu âm thanh đầu vào. 
    # Việc sử dụng biến in_flag giúp tăng độ chính xác của thông tin tần số trung bình tính được, 
    # do mỗi phần tử của mẫu âm thanh sẽ được chia thành nhiều phần hơn để tính trung bình
    in_flag = 256
    for i in range(0, len(frequency_preprocessed)):
        single_sample_processed = []
        if in_flag == 256:
            arr_len = len(frequency_preprocessed[i])
            for j in range(in_flag):
                start = j * arr_len // in_flag
                end = (j + 1) * arr_len // in_flag
                single_sample_processed.append(np.mean(frequency_preprocessed[i][start:end]))
        all_samples_processed.append(single_sample_processed)
    return all_samples_processed




# Đường dẩn đến thư mục chứa file audio 
# all_test_audios, y = custom_database_import("wav")
# X = np.array(custom_preprocess(all_test_audios))
# update 4/2/2023
all_test_audios, y = custom_database_import(args.audio_path)
X = np.array(custom_preprocess(all_test_audios))




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
hyp_parameters = {
    'random_state': [0],
    'n_estimators': [100, 1000],
    'max_depth': [None, 2, 4],
    'max_features': ['auto', 'sqrt']
}
config_cnt = 0
tot_config = 2 * 3 * 2
max_f1 = 0

for config in ParameterGrid(hyp_parameters):
    config_cnt += 1
    print(f'Analizing config {config_cnt} of {tot_config} || Config: {config}')

    clf = RandomForestClassifier(**config)
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)
    acc = accuracy_score(y_test_pred, y_test)
    p1, r1, f11, s1 = precision_recall_fscore_support(y_test, y_test_pred)
    macro_f1 = f11.mean()

    if macro_f1 > max_f1:
        max_f1 = macro_f1
        print(f"-----> Score: {macro_f1}")
        print()


skf  =StratifiedKFold(n_splits=5)
skf.get_n_splits(X,y)
print(X.shape)
i=1
# file_name = 'result/fg1.txt'
# f_log = open(file_name,"w",encoding="utf-8") 
# update 4/2/2023 at 80s icafe
# file_name = args.result_file
# f_log = open(file_name,"w",encoding="utf-8")
file_txt = strftime("%Y-%m-%d_%Hg%Mp%Ss.txt", gmtime())
full_path_txt = os.path.join(args.result_file, file_txt)
f_log = open(full_path_txt, "w", encoding="utf-8")

for train_index, test_index, in skf.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train,y_test = y[train_index], y[test_index]
    logistic_cl.fit(X_train,y_train)
    print('Lần lặp' + str(i), file = f_log)

    y_train_pred =logistic_cl.predict(X_train)
    print('Kết quả trên tập huấn luyện bằng = ' , file = f_log)
    print(classification_report(y_train, y_train_pred , digits=5), file = f_log) 

    print('Kết quả logistic_cl tập Test = ', file = f_log)
    y_test_pred = logistic_cl.predict(X_test)
    print(classification_report(y_test, y_test_pred, digits=5 ), file = f_log)  
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    plt.matshow(cm)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('Gia trị thực')
    plt.xlabel('Gía trị dự đoán')
    plt.show()
    i=i+1



# Đường dẩn đến thư mục cần đánh giá
# all_eval_audios, y_eval = custom_database_import("eval")
#update 4/2/2023 at 80s icafe
eval_dir = args.eval_dir
all_eval_audios, y_eval = custom_database_import(eval_dir)

X_eval  = np.array(custom_preprocess(all_eval_audios))

forest_clf = RandomForestClassifier(max_depth=None, n_estimators=1000)
forest_clf.fit(X, y)
forest_y_final_pred = forest_clf.predict(X_eval)

MLP_clf = MLPClassifier(activation='relu', alpha=0.001, hidden_layer_sizes=1000)
MLP_clf.fit(X, y)
MLP_y_final_pred = MLP_clf.predict(X_eval)
# print(X.shape)
# print(y.shape)
# print(MLP_y_final_pred.shape)

end_time = time.time()
thoigianchay = end_time - start_time
print('Thoi gian thuc thi: ' + str(thoigianchay) + ' giay, trung binh:'+str(thoigianchay/i)+' giay/lan lap' , file = f_log)
f_log.close()
print(classification_report(y_eval, MLP_y_final_pred))
result = str(MLP_y_final_pred) # Convert MLP_y_final_pred to a string using the str() function
print("Kết quả của " + str(y_eval) +" là "+ result)
# print(MLP_y_final_pred)
custom_csv_print(forest_y_final_pred, 'forest_out')
custom_csv_print(MLP_y_final_pred, 'MLP_out')


