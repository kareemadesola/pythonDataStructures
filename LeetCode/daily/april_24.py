def lengthOfLastWord(s: str) -> int:
    return len(s.strip().split(' ')[-1])


def lengthOfLastWordAlt(s: str) -> int:
    i = len(s) - 1
    while not s[i]:
        i -= 1
    end = i

    while i >= 0 and s[i]:
        i -= 1
    start = i
    return end - start


def isIsomorphic(s: str, t: str) -> bool:
    s_to_t, t_to_s = {}, {}
    n = len(s)
    for i in range(n):
        if s[i] in s_to_t and s_to_t[s[i]] != t[i] \
                or t[i] in t_to_s and t_to_s[t[i]] != s[i]:
            return False
        s_to_t[s[i]] = t[i]
        t_to_s[t[i]] = s[i]
    return True
