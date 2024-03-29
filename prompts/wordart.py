PROMPT = """Think step by step to carry out the instruction.

Instruction: Generate the wordart for the giving Char 'P' from the concept of  'World Peace'
Program:
PROMPT0=PROMPTEXTENSION(prompt='World Peace')
GLYPH0=WORDARTSEMANTIC(char='P', font='kaiti', steps=200, prompt=PROMPT0)
TMODEL0=MODELSELECTION(prompt='World Peace')
IMAGE0=WordArtTexture(model=TMODEL0, cond=GLYPH0,  prompt=PROMPT0)
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Generate the wordart for the giving Char 'B' from the concept of ' Bear'.
Program:
PROMPT0=PROMPTEXTENSION(prompt='Bear')
GLYPH0=WORDARTSEMANTIC(char='B', font='kaiti', steps=200, prompt=PROMPT0)
TMODEL0=MODELSELECTION(prompt='World Peace')
IMAGE0=WORDARTTEXTURE(model=TMODEL0, cond=GLYPH0,  prompt=PROMPT0)
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: Generate the wordart for the giving Char 'D' in Vangoh painting style.
Program:
PROMPT0=PROMPTEXTENSION(prompt='Vangoh painting style')
GLYPH0=WORDARTSEMANTIC(char='D', font='kaiti', steps=0, prompt=PROMPT0)
IMAGE0=WORDARTTEXTURE(model='universal', cond=GLYPH0,  prompt=PROMPT0)
FINAL_RESULT=RESULT(var=IMAGE0)

Instruction: {instruction}
Program:
"""

MODEL_SELECTION_PROMPT="""Think step by step to carry out the instruction.
Instruction: Select the proper model for image generation from the following model candidants:
1. MODELID: '3D'; MODELDESC: for the 3D 
2. MODELID: 'CATOON'; MODELDESC: generate the catoon style person potrait 
3. MODELID: 'Painting'; MODELDESC: generate the catoon style person potrait 
"""