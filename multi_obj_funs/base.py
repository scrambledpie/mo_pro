import abc


class BaseFun(abc.ABC):
    """
    A base class for all objective functions. All objective functions must 
    expose their expected input size as well as thei number of output
    objectives and must have a __call__ method.
    """
    _amino_acids = "RHKDESTNQCUGPAVILMFYW"

    def _validate_x(self, x: str):
        """ Check any incomming protein string for length and invalid chars """
        assert x is not None, "cannot evaluate 'None' string"
        assert isinstance(x, str), f"expected type 'str' but got {type(x)}"
        assert len(x) == self.input_length, f"expected input len {self.input_length}, but got {len(x)}"
        for aa in x:
            assert aa in self._amino_acids, f"unexpected character {aa}"

    @abc.abstractmethod
    def __call__(self, x: str):
        """ all objective functions must have a __call__ method """
        pass

    @property
    @abc.abstractmethod
    def input_length(self):
        """ all objective functions must state their input size """
        pass

    @property
    @abc.abstractmethod
    def num_objectives(self):
        """ all objective functions must state their number of objectives """
        pass
