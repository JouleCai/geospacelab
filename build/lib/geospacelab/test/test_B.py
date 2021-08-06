

def func_1(a, b):
    print(a)
    print(a+b)


def func_2(a, b, c):
    func_1(a, b)
    print(c)


class B:
    def __init__(self, value):
        self.c = value
        self.d = value * 2

    def test(self, value):
        self.c = value*3


class A(B):
    def __init__(self, value):
        self.var = None
        self.var = self.test(value)

    @property
    def var(self):
        return self._var

    @var.setter
    def var(self, value):
        if value is None:
            self._var = None
        else:
            self._var = B(value)

