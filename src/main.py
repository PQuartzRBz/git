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

    def get_department(self):
        raise NotImplementedError


class RAI(Student):
    def __init__(self, first, last, score):
        super().__init__(first, last, score)
        self.department = 'RAI'

    def get_department(self):
        return self.department
    
class EE(Student):
    def __init__(self, first, last, score):
        super().__init__(first, last, score)
        self.department = 'EE'
    
    def get_department(self):
        return self.department

class ME(Student):
    def __init__(self, first, last, score):
        super().__init__(first, last, score)
        self.department = 'ME'
    
    def get_department(self):
        return self.department

std1 = RAI('Passakorn','Pattarapakorn',10)
std2 = EE('Peerawich','Pattarapakorn',25)
std3 = ME('Kuntara','Pattarapakorn',25)
std1.apply_raise()
print(std1())
print(std1 + std2)
print(std1.get_department())
print(std2.department)
print(std3.department)