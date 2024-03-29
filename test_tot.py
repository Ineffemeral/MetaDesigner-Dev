# -*- coding: utf-8 -*-
# @Author: lee.lcy
# @Date:   2024-02-08 14:21:08
# @Last Modified by:   lee.lcy
# @Last Modified time: 2024-02-08 22:45:15
import cv2
import os
import os.path as osp
import sys
sys.path.insert(0, './')
os.environ["LLM_API_KEY"]="sk-O11fUN2Zbi5OSgWVBaBf2eAf71034bB3BaC927488b64F744"
os.environ["LLM_API_BASE"]="https://ai-yyds.com/v1"

from plugin.wordart_texture import TreeOfModelsChat



if __name__ == '__main__':
    
    tom = TreeOfModelsChat(
        model_data_path='./data/model_data_sd15.json',
        model_tree_path='./data/model_tree_tot_sd15.json',
    )

    # Step 1: build tree of models
    print("Begin to build a model tree.")
    tom.build_tree_model()
    print("Finish building a model tree.")

    # Step 2: search appropriate model for specific input prompt
    prompt = "a movie, a superman"
    # prompt = "a flower, realistic"
    selected_model = tom.search_model_tree(prompt)
    print(f'Selected models:\n{selected_model}')