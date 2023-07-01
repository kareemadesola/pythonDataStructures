class SeatManager:
    def __init__(self, n):
        self.seat = [False] * n

    def reserve(self):
        for i in range(len(self.seat)):
            if not self.seat[i]:
                self.seat[i] = True
                break

    def un_reserve(self, num):
        self.seat[num - 1] = False
