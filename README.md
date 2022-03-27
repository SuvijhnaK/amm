pip install heuristicsearch

from heuristicsearch.ao_star import AOStar

print("Graph - 1")
heuristic ={'A':1,'B':6,'C':12,'D':10,'E':4,'F':4,'G':5,'H':7}

adjacency_list = {
    'A' : [[('B',1),('C',1)],[('D',1)]],
    'B' : [[('G',1),('H',1)]],
    #'C':[[('J',1)]],
    'D' : [[('E',1),('F',1)]],
    #'G':[[('I',1)]]
}

graph = AOStar(adjacency_list, heuristic,'A')
graph.applyAOStar()

pip install decision-tree-ID3-Algorithm

from decisiontree.ID3Algorithm import ID3
import csv

def load_csv(filename):
    lines=csv.reader(open(filename,"r"))
    dataset = list(lines) 
    headers = dataset.pop(0) 
    return dataset,headers

dataset_train, headers_train = load_csv("./datasets/P4_train.csv")
dataset_test, headers_test = load_csv("./datasets/P4_test.csv")

id3 = ID3(dataset_train,headers_train,dataset_test,headers_test)
id3.build_tree()
id3.classify()
