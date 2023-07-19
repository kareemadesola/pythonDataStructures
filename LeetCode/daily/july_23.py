import bisect
import collections
from typing import List, Optional, Dict

from LeetCode.Biweekly.contest_82 import TreeNode
from LeetCode.explore.linked_list import ListNode


def distributeCookies(cookies: List[int], k: int) -> int:
    cur = [0] * k
    n = len(cookies)

    def dfs(i, zero_count):
        if n - i < zero_count:
            return float('inf')
        if i == n:
            return max(cur)

        res = float('inf')
        for j in range(k):
            zero_count -= int(cur[j] == 0)
            cur[j] += cookies[i]

            res = min(res, dfs(i + 1, zero_count))

            cur[j] -= cookies[i]
            zero_count += int(cur[j] == 0)
        return res

    return dfs(0, k)


def buddyStrings(s: str, goal: str) -> bool:
    if len(s) != len(goal):
        return False
    if sorted(s) != sorted(goal):
        return False
    if s == goal and len(set(s)) < len(goal):
        return True
    dif = [(i, j) for i, j in zip(s, goal) if i != j]
    return len(dif) == 2


def singleNumber(nums: List[int]) -> int:
    cnt = collections.Counter(nums)
    for k in cnt:
        if cnt[k] == 1: return k


def longestSubarray(nums: List[int]) -> int:
    k = 1
    i = 0
    for j in range(len(nums)):
        k -= nums[j] == 0
        if k < 0:
            k += nums[i] == 0
            i += 1
    return j - i


def minSubArrayLen(target: int, nums: List[int]) -> int:
    start = 0
    tmp = 0
    res = len(nums) + 1
    for end in range(len(nums)):
        tmp += nums[end]
        while tmp >= target:
            res = min(res, end - start + 1)
            tmp -= nums[start]
            start += 1
    return res if res <= len(nums) else 0


def maxConsecutiveAnswers(answerKey: str, k: int) -> int:
    def check(chances, char) -> int:
        start = 0
        res = 0
        for end in range(len(answerKey)):
            if answerKey[end] == char:
                chances -= 1
                while chances < 0:
                    if answerKey[start] == char:
                        chances += 1
                    start += 1
            res = max(res, end - start + 1)
        return res

    return max(check(k, "F"), check(k, 'T'))


def putMarbles(weights: List[int], k: int) -> int:
    n = len(weights)
    pair_weights = [0] * (n - 1)
    for i in range(n - 1):
        pair_weights[i] = weights[i] + weights[i + 1]
    pair_weights.sort()

    res = 0
    for i in range(k - 1):
        res += pair_weights[n - 2 - i] - pair_weights[i]
    return res


def kadane_algorithm(nums) -> int:
    max_so_far = nums[0]
    max_ending_here = 0

    for num in nums:
        max_ending_here = max(max_ending_here, 0)
        max_ending_here += num
        max_so_far = max(max_so_far, max_ending_here)

    return max_so_far


def largestVariance(s: str) -> int:
    global_max = 0

    cnt = [0] * 26
    for i in s:
        cnt[ord(i) - ord('a')] += 1

    for i in range(26):
        major = chr(ord('a') + i)
        for j in range(26):

            if i == j or cnt[i] == 0 or cnt[j] == 0:
                continue

            minor = chr(ord('a') + j)
            major_count = 0
            minor_count = 0

            rest_minor = cnt[j]

            for char in s:
                if minor_count > major_count and rest_minor:
                    major_count = 0
                    minor_count = 0

                if char == major:
                    major_count += 1

                elif char == minor:
                    minor_count += 1
                    rest_minor -= 1

                if minor_count > 0:
                    global_max = max(global_max, major_count - minor_count)

    return global_max


def minDepthBFS(root: Optional[TreeNode]) -> int:
    res = 0
    if not root: return res
    q = collections.deque([root])

    while q:
        for _ in range(len(q)):
            curr = q.popleft()
            if curr and not curr.left and not curr.right:
                return res + 1
            if curr.left: q.append(curr.left)
            if curr.right: q.append(curr.right)
        res += 1


def minDepth(root: Optional[TreeNode]) -> int:
    def dfs(node: Optional[TreeNode]) -> int:
        if not node:
            return 0
        if not node.left:
            return 1 + dfs(node.right)
        if not node.right:
            return 1 + dfs(node.left)
        return 1 + min(dfs(node.left), dfs(node.right))

    return dfs(root)


def distanceK(root: TreeNode, target: TreeNode, k: int) -> List[int]:
    graph = collections.defaultdict(list)

    def build_graph(curr: TreeNode, parent: Optional[TreeNode]):
        if parent:
            graph[curr.val].append(parent.val)
            graph[parent.val].append(curr.val)
        if curr.left:
            build_graph(curr.left, curr)
        if curr.right:
            build_graph(curr.right, curr)

    build_graph(root, None)
    visited = {target.val}

    q = collections.deque([target.val])
    while q:
        if k == 0:
            return list(q)

        for _ in range(len(q)):
            cur = q.popleft()
            for neighbour in graph[cur]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    q.append(neighbour)

        k -= 1
    return []


def eventualSafeNodes(graph: List[List[int]]) -> List[int]:
    safe = {}

    def dfs(curr: int) -> bool:
        if curr in safe:
            return safe[curr]
        safe[curr] = False
        for nei in graph[curr]:
            if not dfs(nei):
                return False
        safe[curr] = True
        return True

    res = []
    for i in range(len(graph)):
        if dfs(i):
            res.append(i)
    return res


def canFinish(numCourses: int, prerequisites: List[List[int]]) -> bool:
    graph = {i: [] for i in range(numCourses)}
    for curr, preq in prerequisites:
        graph[curr].append(preq)

    visited = set()

    def dfs(i: int) -> bool:
        if i in visited:
            return False
        if not graph[curr]:
            return True

        visited.add(i)
        for nei in graph[i]:
            if not dfs(nei):
                return False
        visited.remove(i)
        graph[i].clear()
        return True

    for curr in graph:
        if not dfs(curr):
            return False
    return True


def longestSubsequence(arr: List[int], difference: int) -> int:
    n = len(arr)
    dp = {}
    res = 1
    for i in range(n):
        if arr[i] - difference in dp:
            dp[arr[i]] = dp[arr[i] - difference] + 1
            res = max(res, dp[arr[i]])
        else:
            dp[arr[i]] = 1
    return res


def maxValue(events: List[List[int]], k: int) -> int:
    events.sort()
    n = len(events)
    starts = [start for start, _, _ in events]
    dp = [[-1] * n for _ in range(k + 1)]

    def dfs(curr: int, count: int) -> int:
        if count == 0 or curr == n:
            return 0
        if dp[count][curr] == -1:
            return dp[count][curr]
        nxt = bisect.bisect(starts, events[curr][1])
        dp[count][curr] = max(dfs(curr + 1, count), dfs(nxt, count - 1) + events[curr][2])
        return dp[count][curr]

    return dfs(0, k)


def maxValueBU(events: List[List[int]], k: int) -> int:
    events.sort()
    n = len(events)
    starts = [start for start, _, _ in events]
    dp = [[0] * (n + 1) for _ in range(k + 1)]
    for count in range(1, k + 1):
        for curr in range(n - 1, -1, -1):
            nxt = bisect.bisect(starts, events[curr][1])
            dp[count][curr] = max(dp[count][curr + 1], dp[count - 1][nxt] + events[curr][2])
    return dp[k][0]


def smallestSufficientTeam(req_skills: List[str], people: List[List[str]]) -> List[int]:
    n = len(people)
    m = len(req_skills)

    skill_id = {}
    for i, skill in enumerate(req_skills):
        skill_id[skill] = i

    skills_mask_of_person = [0] * n
    for i in range(n):
        for skill in people[i]:
            skills_mask_of_person[i] |= 1 << skill_id[skill]

    dp = [(1 << n) - 1] * (1 << m)
    dp[0] = 0
    for skills_mask in range(1, 1 << m):
        for i in range(n):
            smaller_skills_mask = skills_mask & ~skills_mask_of_person[i]
            if smaller_skills_mask != skills_mask:
                people_mask = dp[smaller_skills_mask] | (1 << i)
                if people_mask.bit_count() < dp[skills_mask].bit_count():
                    dp[skills_mask] = people_mask

    answer_mask = dp[(1 << m) - 1]
    res = []
    for i in range(n):
        if (answer_mask >> i) & 1:
            res.append(i)

    return res


def addTwoNumbers(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    def ll_to_int(l: ListNode) -> int:
        ans = []
        while l:
            ans.append(str(l.val))
            l = l.next
        return int(''.join(ans))

    def int_to_ll(val: int) -> ListNode:
        head = l = ListNode()
        for digit in str(val):
            l.next = ListNode(digit)
            l = l.next
        return head.next

    res = ll_to_int(l1) + ll_to_int(l2)
    return int_to_ll(res)


class LRUCache:
    class Node:
        def __init__(self, key=0, val=0):
            self.prev: Optional[LRUCache.Node] = None
            self.nxt: Optional[LRUCache.Node] = None
            self.key, self.val = key, val

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data: Dict[int, LRUCache.Node] = {}
        self.left, self.right = self.Node(), self.Node()
        self.left.next, self.right.prev = self.right, self.left

    def get(self, key: int) -> int:

        if key in self.data:
            self.remove_node(self.data[key])
            self.insert_node(self.data[key])
            return self.data[key].val
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.data:
            self.remove_node(self.data[key])
        self.data[key] = self.Node(key, value)
        self.insert_node(self.data[key])

        if len(self.data) > self.capacity:
            lru = self.left.nxt
            self.remove_node(lru)
            del self.data[lru.key]

    def insert_node(self, node: Node):
        prev, nxt = self.right.prev, self.right
        prev.nxt, node.prev = node, prev
        node.nxt, nxt.prev = nxt, node

    def remove_node(self, node: Node):
        prev, nxt = node.prev, node.nxt
        prev.nxt, nxt.prev = nxt, prev


def eraseOverlapIntervals(intervals: List[List[int]]) -> int:
    intervals.sort()
    end = intervals[0][1]
    res = 0
    for i in range(1, len(intervals)):
        if intervals[i][0] < end:
            end = min(end, intervals[i][1])
            res += 1
        else:
            end = intervals[i][1]
    return res
