from moviepy.editor import *
import os

# đường dẫn đến thư mục chứa tệp MP4
input_dir = './'

# đường dẫn đến thư mục chứa tệp WAV đầu ra
output_dir = './wav/'

# lặp qua tất cả các tệp trong thư mục đầu vào
for filename in os.listdir(input_dir):
    if filename.endswith('.mp4'):
        # đường dẫn đầy đủ đến tệp đầu vào
        input_path = os.path.join(input_dir, filename)
        
        # đường dẫn đầy đủ đến tệp đầu ra
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.wav')
        
        # chuyển đổi tệp MP4 thành tệp WAV
        video = VideoFileClip(input_path)
        audio = video.audio
        audio.write_audiofile(output_path)