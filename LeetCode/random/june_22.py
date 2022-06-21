# 2022-06-20, Mon, 16:10
# Implement Trie (Prefix Tree)
class TrieNode:
    def __init__(self):
        self.children = {}
        self.end_of_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        curr = self.root
        for i in word:
            if i not in curr.children:
                curr.children[i] = TrieNode()
            curr = curr.children[i]
        curr.end_of_word = True

    def search(self, word: str) -> bool:
        node = self.search_helper(word)
        return node and node.end_of_word

    def starts_with(self, prefix: str) -> bool:
        return bool(self.search_helper(prefix))

    def search_helper(self, word: str):
        curr = self.root
        for i in word:
            if i not in curr.children:
                return
            curr = curr.children[i]
        return curr
