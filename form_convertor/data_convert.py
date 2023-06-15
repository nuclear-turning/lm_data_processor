from asyncio import shield
import random
import pandas as pd
import json
def read_csv(csv_path):
    data = pd.read_csv(csv_path)
    return data.values.tolist()

def read_json(json_path):
    data = json.load(open(json_path,encoding='utf8'))
    return data

import os


prompts = [
    "请简要描述一下提供的图片。\n<image>",
    "精简地描述这张图片。\n<image>",
    "为给定的图片提供简短的描述。\n<image>",
    "为呈现的图片提供简洁的解释。\n<image>",
    "总结图片的视觉内容。\n<image>",
    "对接下来的图片给出简短明了的解释。\n<image>",
    "分享对提供的图片的简洁解读。\n<image>",
    "提供图片关键特征的简洁描述。\n<image>",
    "传达对展示的图片的简洁明了的描述。\n<image>",
    "提供照片的清晰且简洁的总结。\n<image>",
    "编写一份简洁但信息丰富的图片总结。\n<image>",
    "创建一份代表给出图像的紧凑叙述。\n<image>",
    
    "<image>\n请简要描述一下提供的图片。",
    "<image>\n精简地描述这张图片。",
    "<image>\n为给定的图片提供简短的描述。",
    "<image>\n为呈现的图片提供简洁的解释。",
    "<image>\n总结图片的视觉内容。",
    "<image>\n对上面的图片给出简短明了的解释。",
    "<image>\n分享对提供的图片的简洁解读。",
    "<image>\n提供图片关键特征的简洁描述。",
    "<image>\n传达对展示的图片的简洁明了的描述。",
    "<image>\n提供照片的清晰且简洁的总结。",
    "<image>\n编写一份简洁但信息丰富的图片总结。",
    "<image>\n创建一份代表给出图像的紧凑叙述。",
]



def convert_wukong(idx,data,image_dir):
    result = []
    for row in data:
        template_json = {
            "id":"",
            "image": "",
            "conversations": [
            {
                "from": "human",
                "value": random.choice(prompts)
            },
            {
                "from": "gpt",
                "value": ""
            }
            ]
        }
        image_path = image_dir+'/'+str(idx)+'.jpg'
        
        if os.path.exists(image_path):
            template_json["id"] = idx
            template_json["image"] = str(idx)+'.jpg'
            template_json["conversations"][-1]["value"] = row[1]
                
            result.append(template_json)
        idx +=1
    return idx,result

def convert_aic(data):
    result = []
    for row in data:
        for caption in [random.choice(row['caption'])]:
            template_json = {
                "id":row["image_id"].split('.')[0],
                "image": row["image_id"],
                "conversations": [
                {
                    "from": "human",
                    "value": random.choice(prompts)
                },
                {
                    "from": "gpt",
                    "value": caption
                }
                ]
            }    
            result.append(template_json)
    return result
def write2json(data,output_path):
    with open(output_path,'w+') as jf:
        json.dump(data,jf,ensure_ascii=False,indent=4)

def load_wukong():
    data_dir = "/home/gpuall/hehx/MLLM/data/caption/wukong_release/"
    image_dir = "/home/gpuall/hehx/MLLM/data/caption/wukong/vis/"
    idx = 0
    output_path = "/home/gpuall/hehx/MLLM/data/caption/wukong/anno/"
    for csv_file in sorted(os.listdir(data_dir))[:1]:
        print(csv_file)
        data = read_csv(data_dir+csv_file)
        idx,result = convert_wukong(idx,data,image_dir)
        print(len(result))
        return result
        # write2json(result,output_path)
def load_aic():

    output_path = "/home/gpuall/hehx/MLLM/data/caption/ai_challenger/caption_chat.json"
    data = read_json("/home/gpuall/hehx/MLLM/data/ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json")
    result = convert_aic(data)
    print(len(result))
    # write2json(result,output_path)
    return result


def merge_images():
    import shutil
    aic_image_dir = "/home/gpuall/hehx/MLLM/data/caption/ai_challenger/images/"
    wukong_image_dir = "/home/gpuall/hehx/MLLM/data/caption/wukong/vis/"
    for image in os.listdir(aic_image_dir):
        shutil.copy(aic_image_dir+image,"/home/gpuall/hehx/MLLM/data/caption/aic_wukong/images/"+image)
    for image in os.listdir(wukong_image_dir):
        shutil.copy(wukong_image_dir+image,"/home/gpuall/hehx/MLLM/data/caption/aic_wukong/images/"+image)

if __name__ == "__main__":
    wukong_out = "/home/gpuall/hehx/MLLM/data/caption/wukong/chat_caption.json"
    result = load_wukong()
    write2json(result,wukong_out)
    # merge_images()