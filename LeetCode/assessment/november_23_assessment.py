def licenseKeyFormatting(s: str, k: int) -> str:
    s = s.upper().replace('-', '')
    if not s:
        return ''
    tmp = []
    res = []
    for char in s[::-1]:
        tmp.append(char)
        if len(tmp) == k:
            res.append(''.join(tmp[::-1]))
            tmp.clear()
    return ''.join(tmp[::-1]) + '-' + '-'.join(res[::-1]) if tmp else '-'.join(res[::-1])


def licenseKeyFormattingAlt(s: str, k: int) -> str:
    res = []
    for i in range(len(s) - 1, -1, -1):
        if s[i] != '-':
            if len(res) % (k + 1) == k:
                res.append('-')
            res.append(s[i])
    return ''.join(res[::-1]).upper()


def lengthLongestPath(s: str) -> int:
    stack = []
    res = 0
    for s in s.split('\n'):
        level = s.rfind('\t') + 1
        length = len(s) - level

        while len(stack) > level:
            stack.pop()

        if '.' in s:
            res = max(res, (stack[-1] if stack else 0) + length)
        else:
            stack.append((stack[-1] if stack else 0) + length + 1)
    return res
