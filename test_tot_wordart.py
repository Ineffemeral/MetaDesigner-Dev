# -*- coding: utf-8 -*-
# @Author: lee.lcy
# @Date:   2024-02-08 14:21:08
# @Last Modified by:   lee.lcy
# @Last Modified time: 2024-02-08 22:45:15
import cv2
import os
import os.path as osp
import sys
import json

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys


model_id = 'MorningsunLee/cv_sd_wordart-texttypo'

sys.path.insert(0, './')
os.environ["LLM_API_KEY"]="sk-O11fUN2Zbi5OSgWVBaBf2eAf71034bB3BaC927488b64F744"
os.environ["LLM_API_BASE"]="https://ai-yyds.com/v1"

from plugin.tot import TreeOfModelsChat


if __name__ == '__main__':

    tom = TreeOfModelsChat(
        model_data_path='./data/wordart_model_data_sd15.json',
        model_tree_path='./data/wordart_model_tree_tot_sd15.json',
    )

    # Step 1: build tree of models
    print("Begin to build a model tree.")
    tom.build_tree_model()
    print("Finish building a model tree.")
    
    with open('data/wordart_model_data_sd15.json', 'r') as f:
        model_data_all = json.load(f)
        model_tags = {model["model_name"].split('.')[0]: model["trigger_words"] for model in model_data_all}
    

    # Step 2: search appropriate model for specific input prompt
    # prompt = "a movie, a superman"
    # prompt = "a flower, realistic"
    # prompt = "brush drawing"

    # prompt = "Close by was a girl, next to a small stream, with mountains in the distance, realistic"
    # prompt = "Close by was a girl, next to a small stream, with mountains in the distance, ink painting"
    # prompt = "Close by was a girl, next to a small stream, with mountains in the distance, anime"

    prompt = "Close by was a girl, next to a small stream, with mountains in the distance"

    # prompt = "A girl is swimming in the pool"

    # prompt = "Palace architecture"
    
    # prompt = "一行白鹭上青天"

    # prompt = "两只白兔在草地上吃草"

    # prompt = "三个人在教堂里面"

    selected_model = tom.search_model_tree(prompt)
    print(f'Selected models:\n{selected_model}')

    model_name = selected_model['model_name'].split('.')[0]

    # rewrite configuration.json
    os.system('rm /mnt/workspace/.cache/modelscope/MorningsunLee/cv_sd_wordart-texttypo/configuration.json')
    json_path = osp.join('./cfgs', model_name+'.json')
    os.system(f'cp {json_path} /mnt/workspace/.cache/modelscope/MorningsunLee/cv_sd_wordart-texttypo/configuration.json')

    inference = pipeline('wordart-generation', model=model_id)

    dst_img_dir = './tmp'
    neg_prompt = ''

    trigger_words = model_tags[model_name][0]
    # add trigger word and some prefix
    prefix = "masterpiece, best quality, High Resolution"
    if trigger_words != 'None':
        prefix = trigger_words + ',' + prefix
    prompt = prefix + "," + prompt

    if not osp.isdir(dst_img_dir):
        os.makedirs(dst_img_dir)

    input_params = {
        "text": {
            "text_content": "Z",
            "font_name": "dongfangdakai",
            "output_image_ratio": "16:9"
        },
        "prompt": prompt,
        "neg_prompt": neg_prompt,
        "image_short_size": 512,
        "image_num": 2,
    }

    output = inference(input_params)
    print(model_name)
    for idx in range(input_params["image_num"]):
        dst_img_path = osp.join(dst_img_dir, model_name + f'_{idx}.jpg')
        cv2.imwrite(dst_img_path, output[OutputKeys.OUTPUT_IMGS][idx])
