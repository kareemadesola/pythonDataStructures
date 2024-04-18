from typing import List, Tuple


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


class MinStack:

    def __init__(self):
        self.data: List[Tuple[int, int]] = []

    def push(self, val: int) -> None:
        if not self.data:
            self.data.append((val, val))
        else:
            self.data.append((val, min(val, self.getMin())))

    def pop(self) -> None:
        self.data.pop()

    def top(self) -> int:
        return self.data[-1][0]

    def getMin(self) -> int:
        return self.data[-1][1]


def evalRPN(tokens: List[str]) -> int:
    stack = []
    for char in tokens:
        if char in '+-*/':
            b, a = stack.pop(), stack.pop()
            if char == '+':
                stack.append(a + b)
            elif char == '-':
                stack.append(a - b)
            elif char == '*':
                stack.append(a * b)
            else:
                stack.append(trunc(a / b))
        else:
            stack.append(int(char))
    return stack[0]
