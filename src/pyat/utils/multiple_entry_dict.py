class MultipleEntryDict:
    """
    A dictionary allowing to store multiple object per key (object are stored as a list)
    """

    def __init__(self):
        self.dico = {}

    def add(self, key, obj):
        known = self.dico.get(key)
        if not known:
            known = []
            self.dico[key] = known
        known.append(obj)

    def get(self, k):
        return self.dico.get(k)

    def keys(self):
        return self.dico.keys()
