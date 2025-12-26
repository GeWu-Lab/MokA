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

def check_mme_eval_data(li):
    # 按图像名称分组
    image_groups = {}
    for item in li:
        image_name = item.split('\t')[0]
        if image_name not in image_groups:
            image_groups[image_name] = []
        image_groups[image_name].append(item)
    
    new_li = []
    # 处理每个图像组
    for image_name, group in image_groups.items():
        # 如果该图像有偶数条记录，全部保留
        if len(group) % 2 == 0:
            new_li.extend(group)
        # 如果该图像有奇数条记录，去掉最后一条
        else:
            new_li.extend(group[:-1])
    
    assert len(new_li) % 2 == 0
    return new_li


def post_processing(response):
    response=response.replace('\n', '')
    response = response.replace('\n', '').replace('不是', 'No').replace('是', 'Yes').replace('否', 'No')
    response = response.lower().replace('true', 'yes').replace('false', 'no')

    return response


def prepare_mme_eval_data(path,name):
    artwork = []
    celebrity = []
    code = []
    color = []
    common = []
    count = []
    exist = []
    landmark = []
    num_cal = []
    ocr = []
    pos = []
    posters = []
    scene = []
    text_trans = []
    with jsonlines.open(path,'r') as f:
        for idx, sample in enumerate(f):
            question = sample['question']
            answer = sample['answer']
            subtask = sample['subtask']
            raw_predict = sample['output'][0]
            image_path = sample['image_path']
            image_name = image_path.split('/')[-1]

            # matches = re.findall(r'<answer>(.*?)</answer>',raw_predict)
            # if(len(matches) < 1):
            #     matches = re.findall(r'<answer>(.*?)<think>',raw_predict)
            # if(len(matches) < 1):
            predict = raw_predict.replace('\n',' ')
            predict = predict.strip().lower()
            if('yes' in predict):
                predict='yes'
            elif('no' or 'not' in predict):
                predict='no'
            else:
                continue
            predict = predict.strip().lower()


            answer = answer.strip().lower()
            
            
            pred_true = image_name + '\t' + question + '\t' + answer + '\t' + predict + '\n'


            if subtask == 'artwork':
                artwork.append(pred_true)
            elif subtask == 'celebrity':
                celebrity.append(pred_true)
            elif subtask == 'code_reasoning':
                code.append(pred_true)
            elif subtask == 'color':
                color.append(pred_true)
            elif subtask == 'commonsense_reasoning':
                common.append(pred_true)
            elif subtask == 'count':
                count.append(pred_true)
            elif subtask == 'existence':
                exist.append(pred_true)
            elif subtask == 'landmark':
                landmark.append(pred_true)
            elif subtask == 'numerical_calculation':
                num_cal.append(pred_true)
            elif subtask == 'OCR':
                ocr.append(pred_true)
            elif subtask == 'position':
                pos.append(pred_true)
            elif subtask == 'posters':
                posters.append(pred_true)
            elif subtask == 'scene':
                scene.append(pred_true)
            elif subtask == 'text_translation':
                text_trans.append(pred_true) 
    
    print(f'{len(artwork)} {len(celebrity)} {len(code)} {len(color)} {len(common)} {len(count)} {len(exist)} {len(landmark)} {len(num_cal)} {len(ocr)} {len(pos)} {len(posters)} {len(scene)} {len(text_trans)}')



    save_dir ='mme_result/'+ name

    print(save_dir)
    os.makedirs(save_dir,exist_ok=True)
    artwork = check_mme_eval_data(artwork)
    with open(f'{save_dir}/artwork.txt','a') as f:
        for item in artwork:
            f.write(item)
    
    celebrity = check_mme_eval_data(celebrity)
    with open(f'{save_dir}/celebrity.txt','a') as f:
        for item in celebrity:
            f.write(item)

    code = check_mme_eval_data(code)
    with open(f'{save_dir}/code_reasoning.txt','a') as f:
        for item in code:
            f.write(item)

    color = check_mme_eval_data(color)
    with open(f'{save_dir}/color.txt','a') as f:
        for item in color:
            f.write(item)
    
    common = check_mme_eval_data(common)
    with open(f'{save_dir}/commonsense_reasoning.txt','a') as f:
        for item in common:
            f.write(item)
    
    count = check_mme_eval_data(count)
    with open(f'{save_dir}/count.txt','a') as f:
        for item in count:
            f.write(item)

    exist = check_mme_eval_data(exist)
    with open(f'{save_dir}/existence.txt','a') as f:
        for item in exist:
            f.write(item)

    landmark = check_mme_eval_data(landmark)
    with open(f'{save_dir}/landmark.txt','a') as f:
        for item in landmark:
            f.write(item)
    
    num_cal = check_mme_eval_data(num_cal)
    with open(f'{save_dir}/numerical_calculation.txt','a') as f:
        for item in num_cal:
            f.write(item)
    
    ocr = check_mme_eval_data(ocr)
    with open(f'{save_dir}/OCR.txt','a') as f:
        for item in ocr:
            f.write(item)

    pos = check_mme_eval_data(pos)
    with open(f'{save_dir}/position.txt','a') as f:
        for item in pos:
            f.write(item)

    posters = check_mme_eval_data(posters)
    with open(f'{save_dir}/posters.txt','a') as f:
        for item in posters:
            f.write(item)
    
    scene = check_mme_eval_data(posters)
    with open(f'{save_dir}/scene.txt','a') as f:
        for item in scene:
            f.write(item)

    text_trans = check_mme_eval_data(text_trans)
    with open(f'{save_dir}/text_translation.txt','a') as f:
        for item in text_trans:
            f.write(item)
    
    return save_dir



def main():


    model_name=['eval_results/MME/final_test']


    all_path=[]

    for model in model_name:
            path=model
            merge_files(path)

            new_path=os.path.join(path,'merged.jsonl')

            save_dir=prepare_mme_eval_data(new_path,model)
            all_path.append(new_path)
    
    ## Run the evaluation script
    import subprocess

    command = ["python", "mme_score.py", "--results_dir", save_dir]

    subprocess.run(command)



if __name__ == "__main__":
    main()
