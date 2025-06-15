import transformers
import torch

transformers.logging.set_verbosity_error()

class pipeline_with_params:
    def __init__(self):
        self.pipe = None
        self.pipe_config = None
        return None
    
    def pipe_init(self,) -> list[transformers.pipeline, dict]:
        return self.pipe, self.pipe_config
    
    def pipe_destroy(self,):
        del self.pipe
        del self.pipe_config
        torch.cuda.empty_cache()



class llama_3_8B_pipeline(pipeline_with_params):
    def __init__(self):
        super().__init__()
        
    def pipe_init(self) -> list:
        self.pipe = transformers.pipeline(
            "text-generation",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            model_kwargs={"torch_dtype": torch.bfloat16},
            device="cuda"
            )

        self.pipe_config = {
            "max_new_tokens":256,
            "eos_token_id":[
                self.pipe.tokenizer.eos_token_id,
                self.pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ],
            "do_sample":True,
            "temperature":1,
            "top_p":1,
            "pad_token_id":self.pipe.tokenizer.eos_token_id
        }
        return self.pipe, self.pipe_config
    

class qwen_2_7B_pipeline(pipeline_with_params):
    def __init__(self):
        super().__init__()

    def pipe_init(self) -> list:
        self.pipe = transformers.pipeline(
            "text-generation", 
            model="Qwen/Qwen2-7B-Instruct",
            model_kwargs={"torch_dtype": "auto"},
            device="cuda",)

        self.pipe_config = {
            "max_new_tokens":256,
            "do_sample":True,
            "temperature":1,
            "top_p":1,
        }
        return self.pipe, self.pipe_config


class qwen_2_1B5_pipeline(pipeline_with_params):
    def __init__(self):
        super().__init__()

    def pipe_init(self,device="cuda") -> list:
        self.pipe = transformers.pipeline(
            "text-generation", 
            model="Qwen/Qwen2-1.5B-Instruct",
            device=device,)

        self.pipe_config = {
            "max_new_tokens":256,
            "do_sample":True,
            "temperature":1,
            "top_p":1,
        }
        return self.pipe, self.pipe_config


class qwen_2_0B5_pipeline(pipeline_with_params):
    def __init__(self):
        super().__init__()

    def pipe_init(self,device="cuda") -> list:
        self.pipe = transformers.pipeline(
            "text-generation", 
            model="Qwen/Qwen2-0.5B-Instruct",
            device=device,)

        self.pipe_config = {
            "max_new_tokens":256,
            "do_sample":True,
            "temperature":1,
            "top_p":1,
        }
        return self.pipe, self.pipe_config
    

class gemma_2_9B_pipeline(pipeline_with_params):
    def __init__(self):
        super().__init__()
    def pipe_init(self,device="cuda") -> list:
        self.pipe = transformers.pipeline(
            "text-generation", 
            model="google/gemma-2-9b-it",
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device,)

        self.pipe_config = {
            "max_new_tokens":256,
            "do_sample":True,
            "temperature":1,
            "top_p":1,
        }
        return self.pipe, self.pipe_config
    

class mistral03_7B_pipeline(pipeline_with_params):
    def __init__(self):
        super().__init__()
    def pipe_init(self,device="cuda") -> list:
        self.pipe = transformers.pipeline(
            "text-generation", 
            model="mistralai/Mistral-7B-Instruct-v0.3",
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device,)

        self.pipe_config = {
            "max_new_tokens":256,
            "do_sample":True,
            "temperature":1,
            "top_p":1,
        }
        return self.pipe, self.pipe_config    



if __name__ == "__main__":
    obj_qwen = qwen_2_7B_pipeline()
    pipe, pipe_config, = obj_qwen.pipe_init()
    pipe("Hello", **pipe_config)
    del obj_qwen, pipe, pipe_config
    torch.cuda.empty_cache()