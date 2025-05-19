from transformers import AutoTokenizer
from langchain_core.prompts import PromptTemplate
from langchain_teddynote.prompts import load_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

def create_ophtimus_chain(task: str):
    if task == "Ophtimus Diagnosis":
        prompt = load_prompt("app/prompts/Ophtimus_diagnosis.yaml", encoding="utf-8")
        hf = HuggingFacePipeline.from_model_id(
            model_id="MinWook1125/Opthimus_MCQA_EQA_CR_5000",
            task="text-generation",
            pipeline_kwargs={"max_new_tokens": 512, "do_sample": True, "temperature": 0.9, "top_p": 0.95},
        )
        llm = ChatHuggingFace(llm=hf)

    elif task == "Ophtimus Q&A":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        chat_template = [
            {"role": "system", "content": "You are an expert ophthalmologist. Please provide accurate and medically sound answers to the user's ophthalmology-related question."},
            {"role": "user", "content": "{instruction}"}
        ]
        prompt_text = tokenizer.apply_chat_template(chat_template, tokenize=False, add_generation_prompt=True)
        prompt = PromptTemplate.from_template(prompt_text)

        hf = HuggingFacePipeline.from_model_id(
            model_id="BaekSeungJu/Ophtimus-Llama-8B",
            task="text-generation",
            pipeline_kwargs={"max_new_tokens": 512, "do_sample": True, "temperature": 0.9, "top_p": 0.95},
        )
        llm = ChatHuggingFace(llm=hf)

    return prompt | llm | StrOutputParser()
