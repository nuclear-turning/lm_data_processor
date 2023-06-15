
from transformers import LlamaForCausalLM,AutoTokenizer
from deltalm.modeling_deltalm import DeltalmForConditionalGeneration
import torch
from tqdm import tqdm
import json
import os
import deepspeed
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

def save_temp(temp):
    for item in temp:
        temp_file = "temp/"+item["id"]+'.json'
        write_json(item,temp_file)
def check_and_clean():
    pass
import copy
def do_trans(e2c_tool,data):
    result = []
    for item in tqdm(data):
        try: 
            # temp_data = []
            temp_file = "temp/"+item['id']+'.json'
            if os.path.exists(temp_file):
                continue
            # item['lang'] = "en"
            # temp_data.append(copy.deepcopy(item))
            convers_raw_text = []
            convers_text = []
            for convers in item["conversations"]:
                raw_text = convers['value'].replace("\n<image>","").replace("<image>\n","")
                convers_raw_text.append(raw_text)
                if "\n" in raw_text: # 分段且过长的文本
                    raw_text = raw_text.split("\n")
                
                trans_result = e2c_tool.trans(raw_text)
                convers_text.append('\n'.join(trans_result))
            for idx,text in enumerate(convers_text):
                item["conversations"][idx]["en_value"] = item["conversations"][idx]["value"]
                item["conversations"][idx]["value"] = item["conversations"][idx]["value"].replace(convers_raw_text[idx],text)
            
                # print(idx,text,item,item["conversations"][idx])
            del trans_result
            # item['lang'] = "zh"
            # temp_data.append(item)
            
            result.append(item)
            if len(result)>1000:
                save_temp(result)
                result=[]
        except Exception as e:
            print(e)
    save_temp(result)
    del result
    # return result


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
    
    e2c_tool = E2CTranslator(model_cls[model_type],trans_models[model_type],tokenizer[model_type])
    data = read_json(json_path)
    do_trans(e2c_tool,data)
    # write_json(data,output_path)

if __name__ == "__main__":
    main()
    