import csv
import json
import glob
import os

annotations = {}
dataFolderPth = "../../data/"
excludeNormal = False
excludeAbnormal = True
outputFileName = "train.json"

csv_annotation_files = [os.path.join(dataFolderPth, 'Annotations/CSV/UCF - Crime Temporal.csv'),
                        os.path.join(dataFolderPth, 'Annotations/CSV/normal_videos.csv')]

categoryCounter = {
    "Abuse": 0,
    "Arrest": 0,
    "Arson": 0,
    "Assault": 0,
    "Banner": 0,
    "Burglary": 0,
    "Explosion": 0,
    "Fighting": 0,
    "MolotovBomb": 0,
    "RoadAccidents": 0,
    "Robbery": 0,
    "Shooting": 0,
    "Shoplifting": 0,
    "Stealing": 0,
    "Training_Normal_Videos_Anomaly": 0,
    "Vandalism": 0

}

counter = 0
for annotation_file in csv_annotation_files:
    first = True
    with open(annotation_file, newline='\n') as csvfile:
        datasheet = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in datasheet:
            if first:  # info row
                first = False
            else:
                j = 0
                temporal = {'start': -1, 'end': -1}

                # banner and molotof filedir were not added during annotation
                if 'banner' in row[0]:
                    # extension is lowercase or uppercase
                    banner_list = glob.glob(
                        os.path.join(dataFolderPth, 'Videos/Banner',
                                     row[0] + '*'))
                    if len(banner_list) == 0:  # some videos are missed
                        continue
                    row[0] = 'Banner/' + banner_list[0].split('/')[-1]
                elif 'molotof' in row[0]:
                    row[0] = 'MolotovBomb/' + row[0]
                video_name = row[0]

                if excludeNormal:
                    if video_name.startswith("Training_Normal_Videos_Anomaly"):
                        continue

                if excludeAbnormal:
                    if not video_name.startswith("Training_Normal_Videos_Anomaly"):
                        continue

                for category in categoryCounter.keys():
                    if category.lower() in video_name.lower():
                        categoryCounter[category] += 1

                counter += 1
                annotations[video_name] = []

                for element in row:
                    if not element:
                        break
                    if not j == 0:
                        if j % 2 == 1:
                            temporal['start'] = int(element)
                        else:
                            temporal['end'] = int(element)
                            annotations[video_name].append(temporal)
                            temporal = {'start': -1, 'end': -1}

                    # print(element)
                    j += 1

print("Total Videos: {}".format(counter))
print(categoryCounter)
with open(outputFileName, 'w') as outfile:
    json.dump(annotations, outfile, sort_keys=True, indent=4)
print('Finish')
