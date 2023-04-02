# n = 8
# for j in range(1, n):
#     print("single_sample_processed.append(np.mean(frequency_preprocessed[i][{} * len(frequency_preprocessed[i]) // {}:{} * len(frequency_preprocessed[i]) // {}]))".format(j,n, j+1,n))

# luu lai thoi gian
from datetime import datetime
from time import gmtime, strftime
import time
## luu lai thoi gian bat dau chay module (s)
start_time = time.time()

today = datetime.today()

time_text = str(strftime("%Y-%m-%d_%Hg%Mp%Ss"))
print(today)
print(time_text)
