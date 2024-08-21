from datetime import datetime


class TimeRecorder:
    def __init__(self) -> None:
        self._start_time_list = list()
        self._end_time_list = list()

    def start_record(self):
        self._start_time_list.append(datetime.now())

    def end_record(self):
        self._end_time_list.append(datetime.now())

    def get_total_duration_miliseconds(self):
        ms_duration = 0
        for i in range(min(len(self._start_time_list), len(self._end_time_list))):
            ms_duration += int(
                (self._end_time_list[i] - self._start_time_list[i]).total_seconds()
                * 1000
            )

        return ms_duration

    def get_avg_duration_miliseconds(self):
        count = min(len(self._start_time_list), len(self._end_time_list))
        if count == 0:
            return 0

        return self.get_total_duration_miliseconds() // count

    def clear_records(self):
        self._start_time_list.clear()
        self._end_time_list.clear()
