import abc

class BaseFun(abc.ABC):
    _amino_acids = list("RHKDESTNQCUGPAVILMFYW")

    def _validate_x(self, x: str):
        for aa in x:
            assert aa in self._amino_acids, f"unexpected character {aa}"

    @abc.abstractmethod
    def __call__(self, x: str):
        pass