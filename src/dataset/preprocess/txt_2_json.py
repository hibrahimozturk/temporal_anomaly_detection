import os
import glob
import json


def listdir_nohidden_ext(AllVideos_Path, ext='*_C.txt'):  # To ignore hidden files
    file_dir_extension = os.path.join(AllVideos_Path, ext)
    for f in glob.glob(file_dir_extension):
        filename = f.split('/')[-1]
        if not filename.startswith('.'):
            yield os.path.basename(f)


def read_test_annotations():
    annotation_path = '../../data/UCF_Crimes/Temporal_Anomaly_Annotation_For_Testing_Videos/Txt_formate' \
                      '/Temporal_Anomaly_Annotation.txt'

    test_split = "../../data/UCF_Crimes/Anomaly_Detection_splits/Anomaly_Test.txt"

    with open(annotation_path) as f:
        contents = f.readlines()
    contents = [x.strip() for x in contents]

    with open(test_split) as f:
        video_paths = f.readlines()
    video_paths = [x.strip() for x in video_paths]

    eliminatedContents = {}
    for path in video_paths:
        for content in contents:
            if content.split()[0].split("/")[-1] in path:
                eliminatedContents[path] = content

    fps = 30  # fps is fixed in ucf crime
    annotations = {}

    for path, line in eliminatedContents.items():
        line_parts = line.split()
        video_filename = line_parts[0]

        # if not path.startswith("Testing_Normal_Videos_Anomaly"):
        #     continue

        # if path.startswith("Testing_Normal_Videos_Anomaly"):
        #     continue

        if not (video_filename in path):
            print('Warning!!!')
            print(video_filename)

        annotations[path] = []

        if int(line_parts[2]) != -1:
            temporal_annotation = {'start': round(int(line_parts[2]) / fps), 'end': round(int(line_parts[3]) / fps)}
            annotations[path].append(temporal_annotation)

        if int(line_parts[4]) != -1:
            temporal_annotation = {'start': round(int(line_parts[4]) / fps), 'end': round(int(line_parts[5]) / fps)}
            annotations[path].append(temporal_annotation)
    return annotations


test_annotations = read_test_annotations()
with open('annotations_test.json', 'w') as outfile:
    json.dump(test_annotations, outfile, sort_keys=True, indent=4)
