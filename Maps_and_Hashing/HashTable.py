"""Write a HashTable class that stores strings
in a hash table, where keys are calculated
using the first two letters of the string."""
from typing import List, Union


class HashTable:

    def __init__(self):
        self.table: List[Union[None, List[str]]] = [None] * 10000

    def store(self, string):
        """Input a string that's stored in
        the table"""
        hash_value = self.calculate_hash_value(string)
        if not self.table[hash_value]:
            self.table[hash_value] = [string]
        else:
            self.table[hash_value].append(string)

    def lookup(self, string):
        """return the hash value if the string
        is already in the table.
        Return -1 otherwise"""
        hash_value = self.calculate_hash_value(string)
        if self.table[hash_value] and string in self.table[hash_value]:
            return hash_value
        return -1

    def calculate_hash_value(self, string):
        """Helper function to calculate a
        hash value from a string."""
        return ord(string[0]) * 100 + ord(string[1])


if __name__ == '__main__':
    # Set up
    hash_table = HashTable()

    # Test calculate_hash_value
    # Should be 8568
    print(hash_table.calculate_hash_value("UDACITY"))

    # Test lookup edge case
    # Should be -1
    print(hash_table.lookup("UDACITY"))

    # Test store
    print(hash_table.store("UDACITY"))
    # Should be 8568
    print(hash_table.lookup("UDACITY"))

    # Test store edge case
    hash_table.store("UDACIOUS")
    # Should be 8568
    print(hash_table.lookup("UDACIOUS"))
