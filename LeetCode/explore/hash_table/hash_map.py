class HashMap:
    def __init__(self):
        self.size = 1000
        self.data = [[] for _ in range(self.size)]

    # amortize worst time O(k)
    def put(self, key: int, value: int) -> None:
        idx = self.hash_function(key)
        if key not in self:
            self.data[idx].append([key, value])
            return
        for i in self.data[idx]:
            if i[0] == key:
                i[1] = value

    # time O(k)
    def get(self, key: int) -> int:
        if key in self:
            idx = self.hash_function(key)
            for i in self.data[idx]:
                if i[0] == key:
                    return i[1]
        return -1

    # worst time search and remove O(k) * O(k)
    def remove(self, key: int) -> None:
        if key in self:
            idx = self.hash_function(key)
            for i in self.data[idx]:
                if i[0] == key:
                    self.data[idx].remove(i)

    # time O(k) k = N/len_ where N is number of elements
    def __contains__(self, key: int):
        idx = self.hash_function(key)
        return key in (item[0] for item in self.data[idx])

    # time O(1)
    def hash_function(self, key: int) -> int:
        return key % self.size
