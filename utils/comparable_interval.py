from interval import interval, inf


class ComparableInterval(interval):
    """
    Represents a mathematical interval over the reals for interval heuristics over PDDL+
    (https://ebooks.iospress.nl/publication/44811)
    Intervals are considered to satisfy a condition if they contain any values that satisfy the condition.
    """
    def __init__(self, *values):
        interval(*values)

    def __eq__(self, other):
        # intersection is not empty
        return other in self or len(self & other) > 0

    def __lt__(self, other):
        if isinstance(other, ComparableInterval):
            if len(self) == 0 or len(other) == 0:
                return False
            else:
                return self[0].inf < other[-1].sup
        else:
            return self[0].inf < other

    def __gt__(self, other):
        if isinstance(other, ComparableInterval):
            if len(self) == 0 or len(other) == 0:
                return False
            else:
                return self[-1].sup > other[0].inf
        else:
            return self[0].sup > other

    def __le__(self, other):
        return self < other or self == other

    def __ge__(self, other):
        return self > other or self == other

    def __round__(self, n=None):
        return self._canonical(self.Component(round(x.inf, n), round(x.sup, n)) for x in self)

    def inverse(self):
        return self._canonical(self.Component(x.inf, x.sup) for x in interval.inverse(self))


if __name__ == '__main__':
    # unit tests
    a = ComparableInterval([2, 3], [60, 70])
    b = ComparableInterval([0.5, 1.5], [4, 5])
    print(a == b)
    print(a < b)
    print(b.inverse())
    c = b / 2
    print(c)
    d = round(a, 2)
    print(d)
    print(a <= c)
    e = a * c
    print(e)
