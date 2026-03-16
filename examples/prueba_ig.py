import os

from ctfzeros.divideconquer import DCCC_inverted_tree

os.chdir(os.path.abspath(".."))

from ctfzeros.prepro import load_and_preprocess

filepath = "./models/synthetic/simple_nparents2_nzr04_zdr05_10.uai"
datapath = "./models/synthetic/simple_nparents2_nzr04_zdr05_10.csv"

model, data, _, _ = load_and_preprocess(filepath)
model

model.draw()

infDCCC = DCCC_inverted_tree(model, data, num_runs=20)
p = infDCCC.prob_sufficiency("X2", "Y")
print(p)


import ctfzeros.divideconquer
ctfzeros.divideconquer.__file__


import numpy.linalg
numpy.linalg.__file__

print()


# for _ in infDCCC.compile_incremental(step_runs=1):
#    p = infDCCC.prob_sufficiency("X2","Y")
#    print(p)
#infDCCC.compile()
#infDCCC.models
#infDCCC.prob_sufficiency("X2","Y")