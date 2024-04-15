def isValid(s: str) -> bool:
    stack = []
    close_to_open = {')': '(', ']': '[', '}': '{'}
    for char in s:
        if char in '([{':
            stack.append(char)
        elif stack and stack[-1] == close_to_open[char]:
            stack.pop()
        else:
            return False
    return not stack
