

class TSum:

    def __init__(self, name, *terms):
        self.name = name
        self.terms = list(terms)
        
    def __str__(self):
        return self.name


class TProd:

    def __init__(self, name, *terms):
        self.name = name
        self.terms = list(terms)
    
    def __str__(self):
        return self.name


class Vec: 

    def __init__(self, name, size):
        self.name = name
        self.size = size
    
    def __str__(self):
        return self.name

