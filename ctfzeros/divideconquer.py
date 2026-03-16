from typing import Callable

import numpy as np
from bcause.inference.causal import CausalObservationalInference
from bcause.inference.causal.multi import CausalMultiInference
from bcause.inference.inference import Inference
from bcause.inference.probabilistic.datainference import LaplaceInference
from bcause.models.cmodel import StructuralCausalModel

from ctfzeros.model_utils import update_exo_probs, get_missing_states
from ctfzeros.scmgenerator.generators import scm_solution_generator
from ctfzeros.scmgenerator_general.general_solution_generator import scm_general_solution_generator

# Define the logger
#log_format = '%(asctime)s|%(levelname)s|%(filename)s: %(message)s'
#log = get_logger(__name__, fmt=log_format)
#logging.getLogger("bcause").setLevel(logging.INFO)

#log.propagate = 0

class DCCC_inverted_tree(CausalMultiInference, CausalObservationalInference):
    def __init__(self, model:StructuralCausalModel, data, causal_inf_fn: Callable = None, interval_result=True, num_runs=None, new_method=True):
        self._data = data
        self._prior_model = model
        self._num_runs = num_runs or float("inf")
        self._model = model
        self.num_generated = 0
        self.new_method = new_method

        Y = [v for v in model.endogenous if len(model.get_edogenous_parents(v)) > 0][0]
        X = sorted([v for v in model.endogenous if len(model.get_edogenous_parents(v)) == 0])

        # Directly calculate Ux from the data (which will not change in this topology)
        infdata = LaplaceInference(data, model.domains)
        self.new_doms = {u: model.domains[u] for u in [model.get_exogenous_parents(x)[0] for x in X]}
        self.new_probs = {model.get_exogenous_parents(x)[0]: infdata.query(x).values for x in X}
        # m = update_exo_probs(model, new_doms, new_probs)

        # Calculate the distribution P(Y|X1,...,Xm)
        y_dist = infdata.query(Y, conditioning=X).reorder(*X,Y).values
        y_dist = np.array(y_dist).reshape((len(y_dist), 1))


        self.Uy = self.model.get_exogenous_parents(Y)[0]

        if not new_method:
            self.scm_generator = scm_solution_generator(
               n_parents=len(X),
               y_dist=y_dist,
               exclude_us=set(get_missing_states(model.factors[Y], self.Uy)),
               solver= len(X) < 3
            )

        else:


            child_domain_size = len(model.domains[Y])
            probabilities = y_dist
            parent_domain_size = np.prod([len(model.domains[x]) for x in X])
            exclude_us = set(get_missing_states(model.factors[Y], self.Uy))

            self.scm_generator = scm_general_solution_generator(child_domain_size, parent_domain_size,
                                                                   child_dist=probabilities, exclude_us=exclude_us, seed=0)



        super().__init__([], causal_inf_fn=causal_inf_fn, interval_result=interval_result, outliers_removal=False)

    def compile(self, *args, **kwargs) -> Inference:
        if self.new_method:
            models = []
            for domUy, pUy in self.scm_generator:
                domUy = [int(s) for s in domUy]
                self.num_generated += 1
                self.new_doms[self.Uy] = list(domUy)
                self.new_probs[self.Uy] = list(pUy)
                models.append(update_exo_probs(self._prior_model, self.new_doms, self.new_probs))
            self.add_models(models)
            return super().compile()


        else:
            models = []
            for domUy, pUy, _ in self.scm_generator:
                self.new_doms[self.Uy] = list(domUy)
                self.new_probs[self.Uy] = list(pUy)
                models.append(update_exo_probs(self._prior_model, self.new_doms, self.new_probs))
                if len(models)>=self._num_runs: break
            self.add_models(models)
            return super().compile()



    def compile_incremental(self, step_runs=1, *args, **kwargs) -> Inference:

       if self.new_method:

            models = []
            for domUy, pUy in self.scm_generator:
                #self.num_generated += num_generated

                domUy = [int(s) for s in domUy] #{v:[int(s) for s in d] for v,d in domUy.items()}
                self.num_generated += 1
                self.new_doms[self.Uy] = list(domUy)
                self.new_probs[self.Uy] = list(pUy)
                models.append(update_exo_probs(self._prior_model, self.new_doms, self.new_probs))
                if len(models)>=step_runs:
                    self.add_models(models)
                    models = []
                    if len(self._models)>=self._num_runs:
                        return super().compile()
                    yield super().compile()

            self.add_models(models)
            models = []
            yield super().compile()

       else:

        #def compile_incremental_old(self, step_runs=1, *args, **kwargs) -> Inference:
            models = []
            for domUy, pUy, num_generated in self.scm_generator:
                self.num_generated += num_generated
                self.new_doms[self.Uy] = list(domUy)
                self.new_probs[self.Uy] = list(pUy)
                models.append(update_exo_probs(self._prior_model, self.new_doms, self.new_probs))
                if len(models)>=step_runs:
                    self.add_models(models)
                    models = []
                    if len(self._models)>=self._num_runs:
                        return super().compile()
                    yield super().compile()

            self.add_models(models)
            models = []
            yield super().compile()

