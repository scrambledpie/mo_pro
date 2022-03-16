from .base import BaseFun


class ToyFun(BaseFun):
    _matching_sequences = ["AB", "QR", "NG"]

    def __call__(self, x: str) -> list:
        self._validate_x(x)

        y_1 = x.count(self._matching_sequences[0])
        y_2 = x.count(self._matching_sequences[1])
        y_3 = x.count(self._matching_sequences[2])

        return y_1, y_2, y_3