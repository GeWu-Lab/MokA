import jsonlines
import re
import numpy as np
from sklearn.metrics import accuracy_score
import argparse
import os


def check(load_path):


    vocab = set()
    with open('/data/users/user/AVE_Dataset/Annotations.txt','r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        event = line.split('&')[0]
        vocab.add(event)
    vocab = list(vocab)
    print(len(vocab),vocab)
    mapping = {}
    mapping['none'] = 0
    for i,event in enumerate(vocab):
        event = event.lower()
        mapping[event] = i+1
    print(mapping)

    path = load_path
    N = 402 * 10
    c = 0
    pre_labels = np.zeros(N)
    real_labels = np.zeros(N)
    nums = 0


    with jsonlines.open(path,'r') as f:
        for idx, sample in enumerate(f):


            answer = sample['output']
            pred = sample['predict']
            matches = re.findall(r'event:(.*?)start_time', answer)
            event = matches[0].strip().lower()
            start_time = int(answer.split()[-2].split(':')[-1])
            end_time = int(answer.split()[-1].split(':')[-1][:-4])
            
            matches = re.findall(r'<event>(.*?)</event>', pred)
            # matches = re.findall(r'event:(.*?)start',pred)
            if len(matches) != 1:
                print('idx: ',idx, 'pred: ', pred)
                continue
            pred_event = matches[0].strip().lower()
            if pred_event not in mapping:
                print('idx: ',idx, 'event not in mapping.')
                continue
            
            matches = re.findall(r'<range>(.*?)</range>', pred)
            if len(matches) != 1:
                print('idx: ',idx, ' pred: ', pred)
                continue
            try:
                pred_start = int(matches[0].strip().split(',')[0])
                pred_end = int(matches[0].strip().split(',')[1])
            except:
                print('idx: ',idx,' exception')
                continue
            
            nums += 1
            for i in range(10):
                if i >= start_time and i <= end_time:
                    try:
                        real_labels[c] = mapping[event]    
                    except:
                        print(c)
                        print(idx)
                        print(answer)
                        print(pred)
                if i >= pred_start and i <= pred_end:
                    pre_labels[c] = mapping[pred_event]
                c += 1


    print('tot: ',idx)
    real_labels = np.array(real_labels)
    pre_labels = np.array(pre_labels)
    print(len(real_labels))
    print(len(pre_labels))
    acc = accuracy_score(real_labels, pre_labels)
    #print('c: ',c, ' nums: ',nums, 'acc: ',acc)

    return nums,acc



def main():

    model_name=['AVE_ft']

    check_point=['checkpoint-153']

    all_path=[]
    nums=[]
    accs=[]

    for model in model_name:
        for ckpt in check_point:
            path=os.path.join('/data/users/user',model,ckpt,'inference_ave/log_route.jsonl')
            num,acc=check(path)

            all_path.append(path)
            nums.append(num)
            accs.append(acc)
        
        print('--------------------------------------------------------------')
    

    for idx, path in enumerate(all_path):
        print(path)
        print(' nums: %d acc: %.2f'% (nums[idx],accs[idx]*100))
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')








if __name__ == "__main__":
    main()