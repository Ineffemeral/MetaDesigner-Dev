import cv2
import os
import os.path as osp
import json
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys



"""
cfgs/AnythingV5[Prt-RE].json

基础模型
{
    "framework":"pytorch",
    "task":"wordart-generation",
    "pipeline":{"type":"wordart-texture"},
    "model":{
        "type":"wordart-texture",
        "base_model_path": "path/of/base_model.safetensors",
        "lora_model_paths": null,
        "lora_model_ratios": null
    },
    "allow_remote":true
}


lora
{
    "framework":"pytorch",
    "task":"wordart-generation",
    "pipeline":{"type":"wordart-texture"},
    "model":{
        "type":"wordart-texture",
        "base_model_path": "path/of/base_model.safetensors",
        "lora_model_paths": ["path/of/lora.safetensors"],
        "lora_model_ratios": [ratio]
    },
    "allow_remote":true
}
"""

####  这里添加模型json
model_jsons = {
                # Base models:
                
                # 'AnythingV5[Prt-RE]': '/MetaDesigner/cfgs/AnythingV5[Prt-RE].json',
                # 'RevAnimated_v122': '/MetaDesigner/cfgs/RevAnimated_v122.json',
                'majiMIX_realistic': '/MetaDesigner/cfgs/majicMIX_realistic.json',
                # 'DreamShaper_XL_Turbo_Lightning': '/MetaDesigner/cfgs/DreamShaper XL Turbo Lightning.json',
                
                # Lora: 
                
                # 'lingjianshan': '/MetaDesigner/cfgs/lingjianshan_lora.json',
                # 'guofeng_shanshui': '/MetaDesigner/cfgs/guofeng_gongbi_shanshui_lora.json',
                # 'shanhaijing': '/MetaDesigner/cfgs/shanhaijing_lora.json'
                # 'miniature': '/MetaDesigner/cfgs/miniature_lora.json'
                # 'village_scenery': '/MetaDesigner/cfgs/village_scenery_lora.json'
                # 'industrial': '/MetaDesigner/cfgs/industrial_3c_lora.json'
                # 'chinese_bronzes': '/MetaDesigner/cfgs/chinese_bronzes_lora.json'
                # 'daily_drawings_lora': '/MetaDesigner/cfgs/daily_drawings_lora.json'
                # 'futuristic mecha': '/MetaDesigner/cfgs/futuristic_mehca_lora.json'
                # 'pencil_sketch': '/MetaDesigner/cfgs/pencil_sketch_lora.json'
                # 'ink_painting': '/MetaDesigner/cfgs/ink_painting_lora.json'
                # 'bronze_relif': '/MetaDesigner/cfgs/bronze_relif_lora.json'
                # 'Gundamhime': '/MetaDesigner/cfgs/Gundamhime.json'
                # 'american_graffiti': '/MetaDesigner/cfgs/american_graffiti_lora.json'
                # 'artefactorylab': '/MetaDesigner/cfgs/artefactorylab_v1_lora.json'
                # 'super_mecha_car': '/MetaDesigner/cfgs/super_mecha_car_lora.json'
                # 'cyberpunk_girl': "/MetaDesigner/cfgs/cyberpunk_girl_lora.json"
                # 'halo_mecha': '/MetaDesigner/cfgs/halo_mecha_lora.json'
                # 'mecha_man': '/MetaDesigner/cfgs/mecha_man_lora.json'
                # 'futuristic_architecture': "/MetaDesigner/cfgs/futuristic_architecture_lora.json"
                # 'metal_texture': "/MetaDesigner/cfgs/metal_texture_lora.json"
                # 'cyberpunk_scenery_lora': '/MetaDesigner/cfgs/cyberpunk_girl_lora.json'
                # 'light_effect_scifi': '/MetaDesigner/cfgs/light_effect_sci_fi_lora.json'
                'furniture': '/MetaDesigner/cfgs/furniture_lora.json'
                }


model_id = 'MorningsunLee/cv_sd_wordart-texttypo'

dst_img_dir = './tmp'

# Prompt lists
prompt_list = [
    'home appliances, High quality, super detailed, a pot on the table, decorative painting, eggs, seasonings'
    ]

neg_prompt_list = [
    'nsfw, ng_deepnegative_v1_75t,badhandv4, (worst quality:2), (low quality:2), (normal quality:2), lowres,watermark, monochrome,nsfw'
    ]

if not osp.isdir(dst_img_dir):
    os.makedirs(dst_img_dir)

# 首次运行需要提前初始化，
# 自动创建.cachee/modelscope/MorningsunLee/cv_sd_wordart-texttypo目录，
# 后续使用可以注释掉
# inference = pipeline('wordart-generation', model=model_id)

for model_name, json_path in model_jsons.items():
    
    with open(json_path, 'r') as f:
        cfg = json.load(f)
    # 路径一般为.cache/modelscope/MorningsunLee/cv_sd_wordart-texttypo
    # 检查路径是否正确
    f = open('/mnt/workspace/.cache/modelscope/MorningsunLee/cv_sd_wordart-texttypo/configuration.json', 'w')
    f.write(json.dumps(cfg, indent=4))
    f.close()

    # 初始化实时检测pipeline
    inference = pipeline('wordart-generation', model=model_id)
    for prompt_id, (p, n_p) in enumerate(zip(prompt_list, neg_prompt_list)):

        # 提供文案，使用内置布局功能
        input_params = {
            "text": {
                "text_content": "ANYWAY",
                "font_name": "dongfangdakai",
                "output_image_ratio": "16:9"
            },
            "prompt": prompt_list[0],
            
            "neg_prompt": neg_prompt_list[0],
            
            "image_short_size": 512,
            "image_num": 3,
        }

        output = inference(input_params)
        target = input_params['text']['text_content']
        im = output[OutputKeys.OUTPUT_IMGS][0]
        dst_img_path = osp.join(dst_img_dir, f"{model_name}_pmt_{prompt_id}_{target}_0.jpg")
        cv2.imwrite(dst_img_path, im)

        im = output[OutputKeys.OUTPUT_IMGS][1]
        dst_img_path = osp.join(dst_img_dir, f"{model_name}_pmt_{prompt_id}_{target}_1.jpg")
        cv2.imwrite(dst_img_path, im)

        im = output[OutputKeys.OUTPUT_IMGS][2]
        dst_img_path = osp.join(dst_img_dir, f"{model_name}_pmt_{prompt_id}_{target}_2.jpg")
        cv2.imwrite(dst_img_path, im)
    