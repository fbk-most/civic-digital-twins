from __future__ import annotations

from dt_model.model.model import Model


class Ensemble:
    def __init__(self, model: Model, scenario, cv_ensemble_size=20):
        # TODO: what if cvs is empty?
        self.model = model
        self.ensemble = {}
        self.size = 1
        for cv in model.cvs:
            if cv in scenario.keys():
                if len(scenario[cv]) == 1:
                    self.ensemble[cv] = scenario[cv]
                else:
                    self.ensemble[cv] = cv.sample(cv_ensemble_size, subset=scenario[cv])
                    self.size *= cv_ensemble_size
            else:
                self.ensemble[cv] = cv.sample(cv_ensemble_size)
                self.size *= cv_ensemble_size

    def __iter__(self):
        self.pos = {k: 0 for k in self.ensemble.keys()}
        self.pos[list(self.ensemble.keys())[0]] = -1
        return self

    def __next__(self):
        for k in self.ensemble.keys():
            self.pos[k] += 1
            if self.pos[k] < len(self.ensemble[k]):
                return {k: self.ensemble[k][self.pos[k]] for k in self.ensemble.keys()}
            self.pos[k] = 0
        raise StopIteration
