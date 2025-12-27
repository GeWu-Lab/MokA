import jsonlines
import re
import os
import json


def merge_files(path):
    # 用于存储合并后的数据
    files=os.listdir(path)
    merged_data = []
    final_save=os.path.join(path,'merged.jsonl')

    # 遍历所有文件
    for file in files:
        if('result' in file):
            name=os.path.join(path,file)
            with open(name, 'r') as f:
                # 读取每一行并解析为 JSON 对象
                for line in f:
                    merged_data.append(json.loads(line))  # 将每一行解析为 JSON 对象并添加到列表中

    # 将合并后的数据写入到一个新的 jsonl 文件
    with open(final_save, 'w') as f:
        for item in merged_data:
            f.write(json.dumps(item) + '\n')  # 将每个 JSON 对象写入一行

    print("合并完成！")




def check(path):

    correct=0
    total=0
    with jsonlines.open(path,'r') as f:
        for idx, sample in enumerate(f):
            total += 1
            answer = sample['answer']
            answer = answer.strip().lower()
            pred = sample['output'][0]
            pred = pred.strip().lower()
            if('a ' in pred):
                pred='(a)'
            elif('b ' in pred):
                pred='(b)'
            elif('c ' in pred):
                pred='(c)'
            elif('d ' in pred):
                pred='(d)'
            elif('e ' in pred):
                pred='(e)'
            else:
                continue

            if(answer in pred):
                correct += 1


    return total,100 * correct / total



def main():

    check_point=['final_test']

    all_path=[]
    nums=[]
    accs=[]

    for ckpt in check_point:
        path=os.path.join('eval_results/MMBench',ckpt)
        merge_files(path)

        new_path=os.path.join(path,'merged.jsonl')

        num,acc=check(new_path)
        all_path.append(new_path)

        nums.append(num)
        accs.append(acc)
    

    for idx, path in enumerate(all_path):
        print(path)
        print(' nums: %d acc:  %.2f %%'% (nums[idx],accs[idx]))








if __name__ == "__main__":
    main()
