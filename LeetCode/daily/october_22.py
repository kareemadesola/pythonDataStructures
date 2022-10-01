def num_decodings_memo(s: str) -> int:
    dp = {len(s): 1}

    def dfs(i: int):
        if i in dp:
            return dp[i]
        if s[i] == '0':
            return 0

        res = dfs(i + 1)
        if i + 1 < len(s) and (s[i] == '1' or s[i] == '2' and s[i + 1] in '0123456'):
            res += dfs(i + 2)
        dp[i] = res
        return res

    return dfs(0)


def num_decodings_dp(s: str) -> int:
    dp = {len(s): 1}
    for i in range(len(s) - 1, -1, -1):
        if s[i] == '0':
            dp[i] = 0
        else:
            dp[i] = dp[i + 1]
        if i + 1 < len(s) and (s[i] == '1' or s[i] == '2' and s[i + 1] in '0123456'):
            dp[i] += dp[i + 2]
    return dp[0]


def test_num_decoding():
    assert num_decodings_memo('1263') == 3
