

from .test_B import func_2


def func_3(a,b,c, func):
    print(func(a,b,c))

print(func_3(1,2,3, func_2))