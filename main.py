# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

class Em:
    class_level_variable = 0
    def __init__(self,f,l,p):
        Em.class_level_variable += 1
        self.f = f
        self.l = l
        self.p = p
        self.fn = f + ' ' + l





class Dv(Em):
    def __init__(self,f,l,p,pr):
        print('EM is used')
        super().__init__('First name: ' + f ,l,p)
        self.pr = pr
        self.fn = f + ' ' + l + ' ' + str(p)

    def cc(self):
        print( self.f + ' '  + self.l + ' ' +  str(self.p) + ' ' +  self.pr )

# class Mg(Em):
#     def __init__(self, f, l, p, em=None):
#         super().__init__(f, l, p)
#         if em == None:
#             self.em = []
#         else:
#             self.em = em
#
#     def remove(self, nm):
#         self.em.remove(nm)
#
#     def adnm(self, nm):
#         self.em.append(nm)


dv1 = Dv('Pradeep', 'Chakravarthi', 4000, 'Python')
# dv2 = Dv('Chaku', 'v', 4000, 'Java')
# mg1 = Mg('Chaku', 'V', 8000, [dv1])
# mg1.adnm(dv2)
# vars(mg1.em[0])


print(dv1.fn)
print(Em.class_level_variable)
dv1.cc()