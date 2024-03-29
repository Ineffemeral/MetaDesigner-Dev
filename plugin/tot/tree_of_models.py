import json
import os
import os.path as osp
import random

import openai


TREE_OF_MODEL_PROMPT_SUBJECT = """ You are an information analyst who can analyze and abstract a set of words to abstract some representation categories.
Below is a template that can represent the abstracted categories in Subject Dimension belonging to concrete noun:

TEMPLATE:
```
Categories:
- [Subject]
- [Subject]
- ...
```

You MUST abstract the categories in a highly abstract manner only from Subject Dimension and ensure the whole number of categories are fewer than 5.
Then, You MUST remove the Style-related categories.

Please output the categories following the format of TEMPLATE. 

Input: {input}

"""

TREE_OF_MODEL_PROMPT_STYLE = """ You are an information analyst who can analyze and summarize a set of words to abstract some representation categories.
Below is a template that can represent the the abstracted categories in Style Dimension:

TEMPLATE:
```
Categories:
- [Style]
- [Style]
- ...
```

You MUST abstract the categories in a highly abstract manner from only Style dimension and ensure the whole number of categories are fewer than 8.

Please output the Categories following the format of TEMPLATE.

Input: {input}

"""

TREE_OF_MODEL_PROMPT_ = """ You are an information analyst who can create a Knowledge Tree according to the input categories.
Below is a knowledge tree template:

TEMPLATE:
```
Knowledge Tree:
- [Subject]
  - [Style]
  - ...
- [Subject]
  - [Style]
  - ...
- [Subject]
- ...
```

You MUST place the each Style category as subcategory under the Subject categories based on whether it can be well matched with a specific subject category to form a reasonable scene.

Please output the categories following the format of TEMPLATE. Note that the output MUST include all the Subject categories and Style category!

Subject Input: {subject}

Style Input: {style}

"""

TREE_OF_MODEL_PROMPT_ADD_MODELS = """ You are an information analyst who can add some input models to an input knowledge tree according to the similarity of the model tags and the categories of the knowledge tree.

You need to place each input model into the appropriate subcategory on the tree, one by one.
You MUST keep the original content of the knowledge tree.  


Please output the final knowledge tree.

Knowledge Tree Input: {tree}

Models Input: {models}

Model Tags Input: {model_tags}

"""

TOT_PROMPTS = """Identify and behave as an experts that are appropriate to select one element from the input list that best matches the input prompt. 

The final selection output MUST be the same as the TEMPLATE:
TEMPLATE:
```
Selected: [the selected word]
```

Input list: {search_list}

Input prompt: {input}
"""


class TreeOfModels(object):
    """A class for managing tree of models"""
    def __init__(self, 
                 llm_api_key=os.environ.get('LLM_API_KEY', ''),
                 llm_api_base=os.environ.get('LLM_API_BASE', ''),
                 model_data_path='./data/model_data_sd15.json',
                 model_tree_path='./data/model_tree_tot_sd15.json',
        ):
        super(TreeOfModels, self).__init__()
        self.llm_api_key = llm_api_key
        self.llm_api_base = llm_api_base
        self.model_data_path = model_data_path
        self.model_tree_path = model_tree_path

        self._init_llm()
        self._load_model_data()

    def _init_llm(self):
        openai.api_key = self.llm_api_key
        openai.api_base = self.llm_api_base

    def _load_model_data(self):
        with open(self.model_data_path, 'r') as f:
            self.model_data_all = json.load(f)

    def llm(self, 
            input_str, 
            model='gpt-3.5-turbo-instruct', # "text-davinci-003" is deprecated, refer to https://platform.openai.com/docs/deprecations, use 'gpt-3.5-turbo-instruct' as replacement
            temperature=0,
            return_usage=False,
            max_tokens=2000): 

        response = openai.Completion.create(
            model=model,
            prompt=input_str,
            temperature=temperature,
            max_tokens=max_tokens
        )
        result = response.choices[0].text.strip()
        usage = response.usage
        if return_usage:
            return result, usage
        else:
            return result

    def build_tree_model(self):
        if not osp.exists(self.model_tree_path):
            tree_model_names = self.build_tree()
            self.tree_model_infos = self.generate_tree_model_infos(tree_model_names=tree_model_names)
        else: # model_tree_path is provided
            self.tree_model_infos = self.generate_tree_model_infos(tree_model_path=self.model_tree_path)

    def build_tree(self):
        """build the tree of base models
            refer to https://github.com/DiffusionGPT/DiffusionGPT
        """
        model_tags = {model["model_name"]: model["tag"] for model in self.model_data_all}
        tags_only = list(model_tags.values()) 
        model_names = list(model_tags.keys())

        prompt1 = TREE_OF_MODEL_PROMPT_SUBJECT.format(input=tags_only)
        response1, u1 = self.llm(prompt1, temperature=0, return_usage=True)
        print("Subject: ", response1)

        prompt2 = TREE_OF_MODEL_PROMPT_STYLE.format(input=tags_only)
        response2, u2 = self.llm(prompt2, temperature=0.8, return_usage=True)
        print("Style: ", response2)

        prompt_tree = TREE_OF_MODEL_PROMPT_.format(style=response2, subject=response1)
        response, u_t = self.llm(prompt_tree, temperature=0, return_usage=True)
        print(response)

        tree = response.split("Knowledge Tree:")[1]

        model_names = [name.split(".")[0] for name in list(model_tags.keys())]

        prompts = TREE_OF_MODEL_PROMPT_ADD_MODELS.format(model_tags=model_tags, tree=tree, models=model_names)
        
        tree_w_models, u_twm = self.llm(prompts, temperature=0, return_usage=True)
        print(tree_w_models)

        output = {}
        tree_list = tree_w_models.split("\n")
        for category in tree_list:
            if category == '':
                continue
            
            if category.startswith("- "):
                current_key = category[2:]
                output[current_key] = {}
            elif category.startswith("  - "):
                next_key = category[4:]
                output[current_key][next_key] = []
            elif category.startswith("    - "):
                output[current_key][next_key].append(category[6:])
        return output

    def generate_tree_model_infos(self, tree_model_names=None, tree_model_path=None):
        assert (tree_model_names is not None) or (tree_model_path is not None)
        if tree_model_path is not None:
            with open(tree_model_path, 'r') as f:
                tree_model_infos = json.load(f)
            return tree_model_infos

        model_all_data = {model["model_name"].split(".")[0]: model for model in self.model_data_all} # remove the suffix

        tree_model_infos = {}
        for cate_name, sub_category in tree_model_names.items():
            cate_name = cate_name.lower()
            temp_category = {}

            for sec_cate_name, sub_sub_cates in sub_category.items():
                sec_cate_name = sec_cate_name.lower()
                temp_model_list = []
                
                for model_name in sub_sub_cates:
                    model_name = model_name.strip()
                    lower_name = model_name[0].lower() + model_name[1:]
                    if model_name in model_all_data:
                        temp_model_list.append(model_all_data[model_name])
                    elif lower_name in model_all_data:
                        temp_model_list.append(model_all_data[lower_name])

                temp_category[sec_cate_name] = temp_model_list

            tree_model_infos[cate_name] = temp_category
        # write in json
        with open(self.model_tree_path, 'w') as f:
            json.dump(tree_model_infos, f, indent=4, ensure_ascii=False)
        return tree_model_infos

    def search_model_tree(self, input_prompt):
        
        if not hasattr(self, 'tree_model_infos'):
            raise Exception('Your should run build_tree_model() before using search_model_tree().')

        search_space = self.tree_model_infos
        search_path = []

        while not isinstance(search_space, list):
            search_list = list(search_space.keys())
            name = self.search_one_matched(input_prompt, search_list)
            search_path.append(name)
            search_space = search_space[name]

        candidate_model_data = {}
        for model in search_space:
            candidate_model_data[model["model_name"]] = model

        model_selected_final = random.choice(list(candidate_model_data.keys()))
        return candidate_model_data[model_selected_final]

    def search_one_matched(self, inputs, search_list):
        
        tot_prompts = TOT_PROMPTS.format(search_list=search_list, input=inputs)

        model_name, u_tot = self.llm(tot_prompts, temperature=0, return_usage=True)
        print(model_name)
        
        model_name = (model_name.split('\n')[-1] if not model_name.endswith('\n') else model_name.split('\n')[-2]).split(':')[-1]
        
        model_name = model_name.strip(' []\'').lower()

        return model_name