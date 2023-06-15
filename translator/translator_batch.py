
from transformers import LlamaForCausalLM,AutoTokenizer
from deltalm.modeling_deltalm import DeltalmForConditionalGeneration
import torch
from tqdm import tqdm
import json
import os
import deepspeed
from load_data import TransDataSet,build_dataloader
from tqdm import tqdm
# What is the setting of the image?
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
# 值得注意的句子：
# The bus in the image is white and red.（红白相间）
# what could be the possible reasons for the man sitting on top of the possessions in the back of the pickup truck?

class E2CTranslator:
    def __init__(self,model_cls,model_path,tokenizer_path):
        model = model_cls.from_pretrained(model_path)
        # Initialize the DeepSpeed-Inference engine
        ds_engine = deepspeed.init_inference(model,
                                        mp_size=2,
                                        dtype=torch.half,
                                        replace_with_kernel_inject=True)
        self.model = ds_engine.module
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model.eval()

        # text = "what could be the possible reasons for the man sitting on top of the possessions in the back of the pickup truck?"


    def trans(self,texts):
        if not isinstance(texts,list):
            texts = [texts]
        with torch.no_grad():
            batch_input = self.tokenizer(texts,return_tensors="pt",padding="longest").to(self.model.device)
            batch_output = self.model.generate(**batch_input)#,max_new_tokens=1024)
            batch_result = self.tokenizer.batch_decode(batch_output,skip_special_tokens=True)
        del batch_output,batch_input
        torch.cuda.empty_cache()

        return batch_result

def read_json(data_path):
    with open(data_path) as jf:
        data = json.load(jf)
    return data
def write_json(data,output_path):
    with open(output_path,'w+') as jf:
        json.dump(data,jf,ensure_ascii=False,indent=4)

def check_and_clean():
    pass
def save_temp(temp):
    for idx in temp:
        temp_file = "temp/"+idx+'.json'
        write_json(temp[idx],temp_file)

import copy
def do_trans(e2c_tool,dataloader):
    result = {}
    for batch in tqdm(dataloader):
        try: 
            texts = batch["values"]
            # temp_data = []
            # item['lang'] = "en"
            # temp_data.append(copy.deepcopy(item))
            
            trans_result = e2c_tool.trans(texts)
            
            for idx,text in enumerate(trans_result):
                img_id = batch["ids"][idx]
                round = batch["rounds"][idx]
                from_ = batch["froms"][idx]
                raw_text = batch["values"][idx].replace("\n<image>","").replace("<image>\n","")
                trans_value = batch["values"][idx].replace(raw_text,text)
                if img_id in result:
                    result[img_id]["conversations"][round] = result[img_id]["conversations"].get(round,[])+ \
                        [{"from":from_,"value":trans_value,"en_value":batch["values"][idx]}]
                else:
                    result[img_id] = {
                        'id':img_id,'image':img_id+".jpg",
                        "conversations":{}
                    }
            del trans_result
        except Exception as e:
            print(e)
    return result


def main():
    randeng_path = "/gemini/data-2/Randeng-Deltalm-362M-En-Zh"
    randeng_tokenizer = "/gemini/data-2/infoxlm-base"
    ziya_path = "/home/gpuall/hehx/PretrainedModels/LanguageModels/FoundationModels/Ziya-LLaMA-13B-v1.1"
    
    json_path = "/gemini/data-1/llava/llava_instruct_150k.json"
    model_type = "randeng"
    trans_models = {
        "randeng":randeng_path,
        "ziya":ziya_path
        }
    model_cls = {
        "randeng":DeltalmForConditionalGeneration,
        "ziya":LlamaForCausalLM
    }
    tokenizer = {
        "randeng":randeng_tokenizer,
        "ziya":ziya_path
    }
    output_path = "output/llava_instruct_150k_zh.json"
    
    trans_dataset = TransDataSet(json_path)
    trans_dataloader = build_dataloader(trans_dataset,batch_size=16)

    e2c_tool = E2CTranslator(model_cls[model_type],trans_models[model_type],tokenizer[model_type])
    result = do_trans(e2c_tool,trans_dataloader)
    save_temp(result)
    # write_json(data,output_path)

if __name__ == "__main__":
    main()
    