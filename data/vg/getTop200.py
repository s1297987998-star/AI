import json
import numpy as np

graph, word = np.load('./data/vg/graph_500_norm.npy'), np.load('./data/vg/vg_500_vector.npy')
graph, word = graph[:200, :200], word[:200]
np.save('./data/vg/graph_200_norm.npy', graph), np.save('./data/vg/vg_200_vector.npy', word)

label = json.load(open('./data/vg/vg_category_500_labels_index.json', 'r'))
for key in label.keys():
    label[key] = filter(lambda x:x<200, label[key])
json.dump(label, open('./data/vg/vg_category_200_labels_index.json', 'w'))
