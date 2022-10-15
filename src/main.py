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
        pass

    