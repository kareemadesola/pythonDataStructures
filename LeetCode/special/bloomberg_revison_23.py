class UndergroundSystem:

    def __init__(self):
        self.check_in = {}
        self.average_time = {}

    def checkIn(self, id: int, stationName: str, t: int) -> None:
        self.check_in[id] = (stationName, t)

    def checkOut(self, id: int, stationName: str, t: int) -> None:
        check_in_station, check_in_time = self.check_in.pop(id)
        if (check_in_station, stationName) not in self.average_time:
            self.average_time[(check_in_station, stationName)] = [0, 0]
        self.average_time[(check_in_station, stationName)][0] += t - check_in_time
        self.average_time[(check_in_station, stationName)][1] += 1

    def getAverageTime(self, startStation: str, endStation: str) -> float:
        time, freq = self.average_time[(startStation, endStation)]
        return time / freq
