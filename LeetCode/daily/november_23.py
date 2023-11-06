def findMode(root: Optional[TreeNode]) -> List[int]:
    def dfs(curr: TreeNode):
        val_to_cnt[curr.val] += 1
        if curr.left:
            dfs(curr.left)
        if curr.right:
            dfs(curr.right)

    val_to_cnt = defaultdict(int)
    dfs(root)
    mode = max(val_to_cnt.values())
    res = []
    for key, val in val_to_cnt.items():
        if val == mode:
            res.append(key)
    return res


def getWinner(arr: List[int], k: int) -> int:
    max_element = max(arr)
    curr = arr[0]
    win_streak = 0
    q = collections.deque(arr[1:])
    while q:
        opponent = q.popleft()
        if curr > opponent:
            win_streak += 1
            q.append(opponent)
        else:
            q.append(curr)
            curr = opponent
            win_streak = 1
        if win_streak == k or curr == max_element:
            return curr


def getWinnerAlt(arr: List[int], k: int) -> int:
    max_element = max(arr)
    curr = arr[0]
    win_streak = 0

    for i in range(1, len(arr)):
        opponent = arr[i]
        if curr > opponent:
            win_streak += 1
        else:
            curr = opponent
            win_streak = 1

        if win_streak == k or curr == max_element:
            return curr
