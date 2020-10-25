import csv
import json

annotations = {}

csv_annotation_files = '../../Dataset/UCF_Crimes/normal_list.txt'
fpw = open('normal_videos.csv', 'w')  

with open(csv_annotation_files) as fp:  
    line = fp.readline()
    while line:
        line = fp.readline()
        if line != "":
            fpw.write("Training_Normal_Videos_Anomaly/"+line.rstrip()+",,,,,,,,,,,," + "\n")
            print("Training_Normal_Videos_Anomaly/"+ line.rstrip() +",,,,,,,,,,,,")
            
fpw.close()
