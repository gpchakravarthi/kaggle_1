# class Basic():
#     def __init__(self, x=10, y=20):
#         self.x = x
#         self.y = y
#
#     def addition(self):
#         return self.x + self.y
#
#     def subtraction(self):
#         return self.y - self.x
#
#     def time_table(self):
#         for i in range(self.x):
#             print('%i * %i = %i' % (i, self.y, self.y * i))
#
#
# # b = Basic()
# # b.addition()
# # b.subtraction()
# # b.time_table()
#
#
# class Nxt:
#     def __init__(self, x, y):
#         self.basic = Basic(x, y)
#         # self.x = x
#         # self.y = y
#
#     def multi(self):
#         return self.basic.x * self.basic.y
#
#
# n = Nxt(10, 40)
# print(n.basic.addition())
# print(n.basic.subtraction()
import seethis
import Regression.example.coolfix as c


def gf(*args, **kwargs):
    print(args)
    print(kwargs)


def ggf(a: list, b: str) -> int:
    print(a, b)
    return 10


# input reaceving
gf('10', 20, 30, a=1000, b=2000)

print(seethis.foo)

print(c.chandan)
