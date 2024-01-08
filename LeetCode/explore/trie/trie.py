from typing import Dict


#
# class TrieNode:
#     N = 26
#
#     def __init__(self):
#         self.children: List[Optional[TrieNode]] = [None for _ in range(TrieNode.N)]
#


class TrieNode:
    def __init__(self):
        self.children: Dict[str, TrieNode] = {}


if __name__ == "__main__":
    root = TrieNode()
    root.children["a"] = TrieNode()
    print(root.children)
