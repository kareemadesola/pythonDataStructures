class UnionFindQuickFind:
    def __init__(self, size: int):
        self.root = [i for i in range(size)]
        self.size = size

    def find(self, x: int) -> int:
        return self.root[x]

    def union(self, x: int, y: int) -> None:
        x_root = self.find(x)
        y_root = self.find(y)
        for i in range(self.size):
            if self.root[i] == y_root:
                self.root[i] = x_root

    def is_connected(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)


class UnionFindQuickUnion:
    def __init__(self, size: int):
        self.root = [i for i in range(size)]
        self.size = size

    def find(self, x: int) -> int:
        while x != self.root[x]:
            x = self.root[x]
        return x

    def union(self, x: int, y: int):
        x_root = self.find(x)
        y_root = self.find(y)

        if x_root != y_root:
            self.root[y_root] = x_root

    def is_connected(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)


class UnionFindRank:
    def __init__(self, size: int):
        self.root = [i for i in range(size)]
        self.rank = [1] * size
        self.size = size

    def find(self, x: int) -> int:
        while x != self.root[x]:
            x = self.root[x]
        return x

    def union(self, x: int, y: int):
        x_root = self.find(x)
        y_root = self.find(y)

        if x_root == y_root:
            return
        if self.rank[x_root] > self.rank[y_root]:
            self.root[y_root] = x_root
        elif self.rank[x_root] < self.rank[y_root]:
            self.root[x_root] = y_root
        else:
            self.root[y_root] = x_root
            self.rank[x_root] += 1
        self.root[y_root] = x_root

    def is_connected(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)

if __name__ == '__main__':
    # uf = UnionFindQuickFind(10)
    # uf = UnionFindQuickUnion(10)
    uf = UnionFindRank(10)
    # 1-2-5-6-7 3-8-9 4
    uf.union(1, 2)
    uf.union(2, 5)
    uf.union(5, 6)
    uf.union(6, 7)
    uf.union(3, 8)
    uf.union(8, 9)
    print(uf.is_connected(1, 5))  # true
    print(uf.is_connected(5, 7))  # true
    print(uf.is_connected(4, 9))  # false
    # 1-2-5-6-7 3-8-9-4
    uf.union(9, 4)
    print(uf.is_connected(4, 9))  # true
