def count_asterisks(s: str) -> int:
    asterisks = 0
    bar = 0
    for i in s:
        if i == '*' and bar % 2 == 0:
            asterisks += 1
        if i == '|':
            bar += 1
    return asterisks
