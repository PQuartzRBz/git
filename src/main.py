from turtle import st


class Student():
    raise_amt = 1.04

    def __init__(self, first, last, score):
        self.first = first
        self.last = last
        self.score = score

    def fullname(self):
        return '{} {}'.format(self.first, self.last)

    def apply_raise(self):
        self.score = (self.score * self.raise_amt)

    def __call__(self):
        return self.fullname()

    def __len__(self):
        return len(self.first)

    def __add__(self, other):
        return self.score + other.score

std1 = Student('Passakorn','Pattarapakorn',10)
std2 = Student('Peerawich','Pattarapakorn',25)
std1.apply_raise()
print(std1())
print(std1 + std2)

class RAI(Student):
    def department(self):
        pass
    