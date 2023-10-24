import collections


class UndergroundSystem:
    def __init__(self):
        self.check_in = {}
        self.distance = collections.defaultdict(list)

    def checkIn(self, id: int, stationName: str, t: int) -> None:
        self.check_in[id] = (stationName, t)

    def checkOut(self, id: int, stationName: str, t: int) -> None:
        check_in_station_name, check_in_time = self.check_in.pop(id)
        self.distance[(check_in_station_name, stationName)].append(t - check_in_time)

    def getAverageTime(self, startStation: str, endStation: str) -> float:
        return sum(self.distance[(startStation, endStation)]) / len(
            self.distance[(startStation, endStation)]
        )


# Your UndergroundSystem object will be instantiated and called as such:
# obj = UndergroundSystem()
# obj.checkIn(id,stationName,t)
# obj.checkOut(id,stationName,t)
# param_3 = obj.getAverageTime(startStation,endStation)
