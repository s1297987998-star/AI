import os
import json
from urllib.request import urlretrieve

with open('image_data.json', 'r') as F:
    metaData = json.load(F)

    imageIds = []
    with open('train_list_500.txt', 'r') as f:
        for line in f.readlines():
            imageIds.append(int(line.strip('\n')[:-4]))
    with open('test_list_500.txt', 'r') as f:
        for line in f.readlines():
            imageIds.append(int(line.strip('\n')[:-4]))

    metaData.sort(key = lambda x : x['image_id'])
    imageIds.sort()

    indexImage = 0
    for indexMetaData in range(len(metaData)):
        if metaData[indexMetaData]['image_id'] == imageIds[indexImage]:
            indexImage+=1
            urlretrieve(metaData[indexMetaData]['url'], os.path.join('/home/dm/space/Datasets/VG_100K/', str(metaData[indexMetaData]['image_id'])+'.jpg'))     
        if indexImage%1000 == 0 and indexImage != 0:
            print('Download {} images...'.format(indexImage))
