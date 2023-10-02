def reverseWords(s: str) -> str:
    return " ".join(word[::-1] for word in s.split(" "))


def winnerOfGame(colors: str) -> bool:
    global_cnt_a = global_cnt_b = 0
    local_cnt_a = local_cnt_b = 0

    for char in colors:
        if char == "A":
            local_cnt_a += 1
        else:
            global_cnt_a += max(local_cnt_a - 2, 0)
            local_cnt_a = 0
    global_cnt_a += max(local_cnt_a - 2, 0)

    for char in colors:
        if char == "B":
            local_cnt_b += 1
        else:
            global_cnt_b += max(local_cnt_b - 2, 0)
            local_cnt_b = 0
    global_cnt_b += max(local_cnt_b - 2, 0)
    return global_cnt_a > global_cnt_b
