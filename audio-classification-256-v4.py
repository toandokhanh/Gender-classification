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




def custom_database_import(in_path):
    # Lấy danh sách các tệp âm thanh trong thư mục đầu vào có đuôi .wav
    index_list = [f for f in os.listdir(in_path) if f.endswith('.wav')]
    
    # Khởi tạo danh sách để chứa các tệp âm thanh và nhãn tương ứng
    in_all_audios = []
    in_y = []
    
    # Duyệt qua danh sách các tệp âm thanh
    for filename in index_list:
#         print(filename)
        # Xây dựng đường dẫn đầy đủ đến tệp âm thanh hiện tại
        full_path = os.path.join(in_path, filename)
        
        # Đọc tệp âm thanh hiện tại và thêm vào danh sách
        rate, data = scipy.io.wavfile.read(full_path, mmap=False)
        in_all_audios.append((rate, data))
        
        # Trích xuất nhãn từ tên tệp và thêm vào danh sách
#         label = filename.split("_")[0] 
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

#     index_list = sorted(index_list, key=lambda x: int(x[:-4]))

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


    # Sampling
    in_flag = 256
    for i in range(0, len(frequency_preprocessed)):
        single_sample_processed = []
        if in_flag == 256:
            single_sample_processed.append(
                np.mean(frequency_preprocessed[i][:1 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 1 * len(frequency_preprocessed[i]) // 256:2 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 2 * len(frequency_preprocessed[i]) // 256:3 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 3 * len(frequency_preprocessed[i]) // 256:4 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 4 * len(frequency_preprocessed[i]) // 256:5 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 5 * len(frequency_preprocessed[i]) // 256:6 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 6 * len(frequency_preprocessed[i]) // 256:7 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 7 * len(frequency_preprocessed[i]) // 256:8 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 8 * len(frequency_preprocessed[i]) // 256:9 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 9 * len(frequency_preprocessed[i]) // 256:10 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 10 * len(frequency_preprocessed[i]) // 256:11 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 11 * len(frequency_preprocessed[i]) // 256:12 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 12 * len(frequency_preprocessed[i]) // 256:13 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 13 * len(frequency_preprocessed[i]) // 256:14 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 14 * len(frequency_preprocessed[i]) // 256:15 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 15 * len(frequency_preprocessed[i]) // 256:16 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 16 * len(frequency_preprocessed[i]) // 256:17 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 17 * len(frequency_preprocessed[i]) // 256:18 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 18 * len(frequency_preprocessed[i]) // 256:19 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 19 * len(frequency_preprocessed[i]) // 256:20 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 20 * len(frequency_preprocessed[i]) // 256:21 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 21 * len(frequency_preprocessed[i]) // 256:22 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 22 * len(frequency_preprocessed[i]) // 256:23 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 23 * len(frequency_preprocessed[i]) // 256:24 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 24 * len(frequency_preprocessed[i]) // 256:25 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 25 * len(frequency_preprocessed[i]) // 256:26 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 26 * len(frequency_preprocessed[i]) // 256:27 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 27 * len(frequency_preprocessed[i]) // 256:28 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 28 * len(frequency_preprocessed[i]) // 256:29 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 29 * len(frequency_preprocessed[i]) // 256:30 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 30 * len(frequency_preprocessed[i]) // 256:31 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 31 * len(frequency_preprocessed[i]) // 256:32 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 32 * len(frequency_preprocessed[i]) // 256:33 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 33 * len(frequency_preprocessed[i]) // 256:34 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 34 * len(frequency_preprocessed[i]) // 256:35 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 35 * len(frequency_preprocessed[i]) // 256:36 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 36 * len(frequency_preprocessed[i]) // 256:37 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 37 * len(frequency_preprocessed[i]) // 256:38 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 38 * len(frequency_preprocessed[i]) // 256:39 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 39 * len(frequency_preprocessed[i]) // 256:40 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 40 * len(frequency_preprocessed[i]) // 256:41 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 41 * len(frequency_preprocessed[i]) // 256:42 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 42 * len(frequency_preprocessed[i]) // 256:43 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 43 * len(frequency_preprocessed[i]) // 256:44 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 44 * len(frequency_preprocessed[i]) // 256:45 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 45 * len(frequency_preprocessed[i]) // 256:46 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 46 * len(frequency_preprocessed[i]) // 256:47 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 47 * len(frequency_preprocessed[i]) // 256:48 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 48 * len(frequency_preprocessed[i]) // 256:49 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 49 * len(frequency_preprocessed[i]) // 256:50 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 50 * len(frequency_preprocessed[i]) // 256:51 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 51 * len(frequency_preprocessed[i]) // 256:52 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 52 * len(frequency_preprocessed[i]) // 256:53 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 53 * len(frequency_preprocessed[i]) // 256:54 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 54 * len(frequency_preprocessed[i]) // 256:55 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 55 * len(frequency_preprocessed[i]) // 256:56 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 56 * len(frequency_preprocessed[i]) // 256:57 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 57 * len(frequency_preprocessed[i]) // 256:58 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 58 * len(frequency_preprocessed[i]) // 256:59 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 59 * len(frequency_preprocessed[i]) // 256:60 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 60 * len(frequency_preprocessed[i]) // 256:61 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 61 * len(frequency_preprocessed[i]) // 256:62 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 62 * len(frequency_preprocessed[i]) // 256:63 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 63 * len(frequency_preprocessed[i]) // 256:64 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 64 * len(frequency_preprocessed[i]) // 256:65 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 65 * len(frequency_preprocessed[i]) // 256:66 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 66 * len(frequency_preprocessed[i]) // 256:67 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 67 * len(frequency_preprocessed[i]) // 256:68 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 68 * len(frequency_preprocessed[i]) // 256:69 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 69 * len(frequency_preprocessed[i]) // 256:70 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 70 * len(frequency_preprocessed[i]) // 256:71 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 71 * len(frequency_preprocessed[i]) // 256:72 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 72 * len(frequency_preprocessed[i]) // 256:73 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 73 * len(frequency_preprocessed[i]) // 256:74 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 74 * len(frequency_preprocessed[i]) // 256:75 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 75 * len(frequency_preprocessed[i]) // 256:76 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 76 * len(frequency_preprocessed[i]) // 256:77 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 77 * len(frequency_preprocessed[i]) // 256:78 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 78 * len(frequency_preprocessed[i]) // 256:79 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 79 * len(frequency_preprocessed[i]) // 256:80 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 80 * len(frequency_preprocessed[i]) // 256:81 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 81 * len(frequency_preprocessed[i]) // 256:82 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 82 * len(frequency_preprocessed[i]) // 256:83 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 83 * len(frequency_preprocessed[i]) // 256:84 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 84 * len(frequency_preprocessed[i]) // 256:85 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 85 * len(frequency_preprocessed[i]) // 256:86 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 86 * len(frequency_preprocessed[i]) // 256:87 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 87 * len(frequency_preprocessed[i]) // 256:88 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 88 * len(frequency_preprocessed[i]) // 256:89 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 89 * len(frequency_preprocessed[i]) // 256:90 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 90 * len(frequency_preprocessed[i]) // 256:91 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 91 * len(frequency_preprocessed[i]) // 256:92 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 92 * len(frequency_preprocessed[i]) // 256:93 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 93 * len(frequency_preprocessed[i]) // 256:94 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 94 * len(frequency_preprocessed[i]) // 256:95 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 95 * len(frequency_preprocessed[i]) // 256:96 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 96 * len(frequency_preprocessed[i]) // 256:97 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 97 * len(frequency_preprocessed[i]) // 256:98 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 98 * len(frequency_preprocessed[i]) // 256:99 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 99 * len(frequency_preprocessed[i]) // 256:100 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 100 * len(frequency_preprocessed[i]) // 256:101 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 101 * len(frequency_preprocessed[i]) // 256:102 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 102 * len(frequency_preprocessed[i]) // 256:103 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 103 * len(frequency_preprocessed[i]) // 256:104 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 104 * len(frequency_preprocessed[i]) // 256:105 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 105 * len(frequency_preprocessed[i]) // 256:106 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 106 * len(frequency_preprocessed[i]) // 256:107 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 107 * len(frequency_preprocessed[i]) // 256:108 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 108 * len(frequency_preprocessed[i]) // 256:109 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 109 * len(frequency_preprocessed[i]) // 256:110 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 110 * len(frequency_preprocessed[i]) // 256:111 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 111 * len(frequency_preprocessed[i]) // 256:112 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 112 * len(frequency_preprocessed[i]) // 256:113 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 113 * len(frequency_preprocessed[i]) // 256:114 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 114 * len(frequency_preprocessed[i]) // 256:115 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 115 * len(frequency_preprocessed[i]) // 256:116 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 116 * len(frequency_preprocessed[i]) // 256:117 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 117 * len(frequency_preprocessed[i]) // 256:118 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 118 * len(frequency_preprocessed[i]) // 256:119 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 119 * len(frequency_preprocessed[i]) // 256:120 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 120 * len(frequency_preprocessed[i]) // 256:121 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 121 * len(frequency_preprocessed[i]) // 256:122 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 122 * len(frequency_preprocessed[i]) // 256:123 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 123 * len(frequency_preprocessed[i]) // 256:124 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 124 * len(frequency_preprocessed[i]) // 256:125 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 125 * len(frequency_preprocessed[i]) // 256:126 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 126 * len(frequency_preprocessed[i]) // 256:127 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 127 * len(frequency_preprocessed[i]) // 256:128 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 128 * len(frequency_preprocessed[i]) // 256:129 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 129 * len(frequency_preprocessed[i]) // 256:130 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 130 * len(frequency_preprocessed[i]) // 256:131 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 131 * len(frequency_preprocessed[i]) // 256:132 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 132 * len(frequency_preprocessed[i]) // 256:133 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 133 * len(frequency_preprocessed[i]) // 256:134 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 134 * len(frequency_preprocessed[i]) // 256:135 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 135 * len(frequency_preprocessed[i]) // 256:136 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 136 * len(frequency_preprocessed[i]) // 256:137 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 137 * len(frequency_preprocessed[i]) // 256:138 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 138 * len(frequency_preprocessed[i]) // 256:139 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 139 * len(frequency_preprocessed[i]) // 256:140 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 140 * len(frequency_preprocessed[i]) // 256:141 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 141 * len(frequency_preprocessed[i]) // 256:142 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 142 * len(frequency_preprocessed[i]) // 256:143 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 143 * len(frequency_preprocessed[i]) // 256:144 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 144 * len(frequency_preprocessed[i]) // 256:145 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 145 * len(frequency_preprocessed[i]) // 256:146 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 146 * len(frequency_preprocessed[i]) // 256:147 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 147 * len(frequency_preprocessed[i]) // 256:148 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 148 * len(frequency_preprocessed[i]) // 256:149 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 149 * len(frequency_preprocessed[i]) // 256:150 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 150 * len(frequency_preprocessed[i]) // 256:151 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 151 * len(frequency_preprocessed[i]) // 256:152 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 152 * len(frequency_preprocessed[i]) // 256:153 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 153 * len(frequency_preprocessed[i]) // 256:154 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 154 * len(frequency_preprocessed[i]) // 256:155 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 155 * len(frequency_preprocessed[i]) // 256:156 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 156 * len(frequency_preprocessed[i]) // 256:157 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 157 * len(frequency_preprocessed[i]) // 256:158 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 158 * len(frequency_preprocessed[i]) // 256:159 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 159 * len(frequency_preprocessed[i]) // 256:160 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 160 * len(frequency_preprocessed[i]) // 256:161 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 161 * len(frequency_preprocessed[i]) // 256:162 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 162 * len(frequency_preprocessed[i]) // 256:163 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 163 * len(frequency_preprocessed[i]) // 256:164 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 164 * len(frequency_preprocessed[i]) // 256:165 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 165 * len(frequency_preprocessed[i]) // 256:166 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 166 * len(frequency_preprocessed[i]) // 256:167 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 167 * len(frequency_preprocessed[i]) // 256:168 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 168 * len(frequency_preprocessed[i]) // 256:169 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 169 * len(frequency_preprocessed[i]) // 256:170 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 170 * len(frequency_preprocessed[i]) // 256:171 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 171 * len(frequency_preprocessed[i]) // 256:172 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 172 * len(frequency_preprocessed[i]) // 256:173 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 173 * len(frequency_preprocessed[i]) // 256:174 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 174 * len(frequency_preprocessed[i]) // 256:175 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 175 * len(frequency_preprocessed[i]) // 256:176 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 176 * len(frequency_preprocessed[i]) // 256:177 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 177 * len(frequency_preprocessed[i]) // 256:178 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 178 * len(frequency_preprocessed[i]) // 256:179 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 179 * len(frequency_preprocessed[i]) // 256:180 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 180 * len(frequency_preprocessed[i]) // 256:181 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 181 * len(frequency_preprocessed[i]) // 256:182 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 182 * len(frequency_preprocessed[i]) // 256:183 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 183 * len(frequency_preprocessed[i]) // 256:184 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 184 * len(frequency_preprocessed[i]) // 256:185 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 185 * len(frequency_preprocessed[i]) // 256:186 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 186 * len(frequency_preprocessed[i]) // 256:187 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 187 * len(frequency_preprocessed[i]) // 256:188 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 188 * len(frequency_preprocessed[i]) // 256:189 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 189 * len(frequency_preprocessed[i]) // 256:190 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 190 * len(frequency_preprocessed[i]) // 256:191 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 191 * len(frequency_preprocessed[i]) // 256:192 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 192 * len(frequency_preprocessed[i]) // 256:193 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 193 * len(frequency_preprocessed[i]) // 256:194 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 194 * len(frequency_preprocessed[i]) // 256:195 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 195 * len(frequency_preprocessed[i]) // 256:196 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 196 * len(frequency_preprocessed[i]) // 256:197 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 197 * len(frequency_preprocessed[i]) // 256:198 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 198 * len(frequency_preprocessed[i]) // 256:199 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 199 * len(frequency_preprocessed[i]) // 256:200 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 200 * len(frequency_preprocessed[i]) // 256:201 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 201 * len(frequency_preprocessed[i]) // 256:202 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 202 * len(frequency_preprocessed[i]) // 256:203 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 203 * len(frequency_preprocessed[i]) // 256:204 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 204 * len(frequency_preprocessed[i]) // 256:205 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 205 * len(frequency_preprocessed[i]) // 256:206 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 206 * len(frequency_preprocessed[i]) // 256:207 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 207 * len(frequency_preprocessed[i]) // 256:208 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 208 * len(frequency_preprocessed[i]) // 256:209 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 209 * len(frequency_preprocessed[i]) // 256:210 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 210 * len(frequency_preprocessed[i]) // 256:211 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 211 * len(frequency_preprocessed[i]) // 256:212 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 212 * len(frequency_preprocessed[i]) // 256:213 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 213 * len(frequency_preprocessed[i]) // 256:214 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 214 * len(frequency_preprocessed[i]) // 256:215 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 215 * len(frequency_preprocessed[i]) // 256:216 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 216 * len(frequency_preprocessed[i]) // 256:217 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 217 * len(frequency_preprocessed[i]) // 256:218 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 218 * len(frequency_preprocessed[i]) // 256:219 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 219 * len(frequency_preprocessed[i]) // 256:220 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 220 * len(frequency_preprocessed[i]) // 256:221 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 221 * len(frequency_preprocessed[i]) // 256:222 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 222 * len(frequency_preprocessed[i]) // 256:223 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 223 * len(frequency_preprocessed[i]) // 256:224 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 224 * len(frequency_preprocessed[i]) // 256:225 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 225 * len(frequency_preprocessed[i]) // 256:226 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 226 * len(frequency_preprocessed[i]) // 256:227 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 227 * len(frequency_preprocessed[i]) // 256:228 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 228 * len(frequency_preprocessed[i]) // 256:229 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 229 * len(frequency_preprocessed[i]) // 256:230 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 230 * len(frequency_preprocessed[i]) // 256:231 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 231 * len(frequency_preprocessed[i]) // 256:232 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 232 * len(frequency_preprocessed[i]) // 256:233 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 233 * len(frequency_preprocessed[i]) // 256:234 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 234 * len(frequency_preprocessed[i]) // 256:235 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 235 * len(frequency_preprocessed[i]) // 256:236 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 236 * len(frequency_preprocessed[i]) // 256:237 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 237 * len(frequency_preprocessed[i]) // 256:238 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 238 * len(frequency_preprocessed[i]) // 256:239 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 239 * len(frequency_preprocessed[i]) // 256:240 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 240 * len(frequency_preprocessed[i]) // 256:241 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 241 * len(frequency_preprocessed[i]) // 256:242 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 242 * len(frequency_preprocessed[i]) // 256:243 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 243 * len(frequency_preprocessed[i]) // 256:244 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 244 * len(frequency_preprocessed[i]) // 256:245 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 245 * len(frequency_preprocessed[i]) // 256:246 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 246 * len(frequency_preprocessed[i]) // 256:247 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 247 * len(frequency_preprocessed[i]) // 256:248 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 248 * len(frequency_preprocessed[i]) // 256:249 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 249 * len(frequency_preprocessed[i]) // 256:250 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 250 * len(frequency_preprocessed[i]) // 256:251 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 251 * len(frequency_preprocessed[i]) // 256:252 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 252 * len(frequency_preprocessed[i]) // 256:253 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 253 * len(frequency_preprocessed[i]) // 256:254 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 254 * len(frequency_preprocessed[i]) // 256:255 * len(frequency_preprocessed[i]) // 256]))
            single_sample_processed.append(np.mean(frequency_preprocessed[i][ 255 * len(frequency_preprocessed[i]) // 256:256 * len(frequency_preprocessed[i]) // 256]))
        all_samples_processed.append(single_sample_processed)

    return all_samples_processed



# Đường dẩn đến thư mục chứa file audio 
all_test_audios, y = custom_database_import("wav")
# dulieu cua file audio
X = np.array(custom_preprocess(all_test_audios))




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# print('X_train.shape')
# print(X_train.shape)
# print('X_test.shape')
# print(X_test.shape)
# print(X)
# print(y)
# print(y.shape)
# print(X.shape)
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
file_name = 'result/fg1.txt'
f_log = open(file_name,"w",encoding="utf-8") 

for train_index, test_index, in skf.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train,y_test = y[train_index], y[test_index]
    logistic_cl.fit(X_train,y_train)
    print('Lần lặp' + str(i), file = f_log)

    y_train_pred =logistic_cl.predict(X_train)
    print('Kết quả trên tập huấn luyện bằng = ' , file = f_log)
    print(classification_report(y_train, y_train_pred , digits=5), file = f_log) 

    print('Kết quả logistic_cl tập Test = ')
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

f_log.close()

# Đường dẩn đến thư mục cần đánh giá
all_eval_audios, y_eval = custom_database_import("eval")

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
# label = filename.split("_")[1] 
print(classification_report(y_eval, MLP_y_final_pred))
result = str(MLP_y_final_pred) # Convert MLP_y_final_pred to a string using the str() function
print("Kết quả của " + str(y_eval) +" là "+ result)
# print(MLP_y_final_pred)
custom_csv_print(forest_y_final_pred, 'forest_out')
custom_csv_print(MLP_y_final_pred, 'MLP_out')


