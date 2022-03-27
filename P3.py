import pandas as pd
import numpy as np

data= pd.read_csv('enjoysport.csv')
concepts= np.array(data.iloc[:,0:-1])
target= np.array(data.iloc[:,-1])
print(concepts)
print(target)

def learn(concepts,target):
    specific_h= concepts[0].copy()
    general_h= [['?' for i in range(len(specific_h))]for i in range(len(specific_h))]
    print("initialization of specific_h and general_h") 
    print(specific_h)
    print(general_h)
    
    for i,h in enumerate(concepts):
        print("For loop begins")
        if target[i]=="yes":
            print("If instance is positive")
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x]='?'
                    general_h[x][x]='?'
                    
        if target[i]=="no":
            print("If instance is negaitive")
            for x in range(len(specific_h)):                  
                if h[x] != specific_h[x]:
                    general_h[x][x]=specific_h[x]
                else:
                    general_h[x][x]='?'
                    
        print("Steps of candidate elemination algorithm",i+1)
        print(specific_h)
        print(general_h)   
        print("\n")
        print("\n")

    indices =[i for i, val in enumerate(general_h) if val==['?','?','?','?','?','?'] ]   
    for i in indices:
        general_h.remove(['?','?','?','?','?','?'])
    return specific_h, general_h
s_final, g_final = learn(concepts, target)
    
print("Final Specific_h:", s_final, sep="\n")
print("Final General_h:", g_final, sep="\n")
 
    
