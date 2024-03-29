import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

os.environ["OPENAI_API_KEY"]="sk-O11fUN2Zbi5OSgWVBaBf2eAf71034bB3BaC927488b64F744"

from PIL import Image
from IPython.core.display import HTML

from engine.utils import ProgramGenerator, ProgramInterpreter
from prompts.wordart import PROMPT

interpreter = ProgramInterpreter(dataset='wordart')

def create_prompt(instruction):
    return PROMPT.format(instruction=instruction)

generator = ProgramGenerator(prompter=create_prompt)
init_state = dict()
instruction = "Generate the wordart for giving char 'S' from the concept of  'snake in the tree'"
# instruction = "Hide Salman and Aamir's faces with :ps, Shahrukh's faces with 8) and Hritik's with ;)"
# instruction = "Create a colorpop of the man in black henley and also blur the background"
prog,_ = generator.generate(instruction)

print(f"prog: {prog}")

result, prog_state = interpreter.execute(prog, init_state, inspect=False)
print(result)