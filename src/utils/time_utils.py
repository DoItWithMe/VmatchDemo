from datetime import datetime


class TimeRecorder:
    def __init__(self) -> None:
        self.start_time_list = list()
        self.end_time_list = list()

    def start_record(self):
        self.start_time_list.append(datetime.now())

    def end_record(self):
        self.end_time_list.append(datetime.now())

    def get_duration_miliseconds(self):
        ms_duration = 0
        for i in range(min(len(self.start_time_list), len(self.end_time_list))):
            ms_duration += int(
                (self.start_time_list[i] - self.end_time_list[i]).total_seconds() * 1000
            )

        return ms_duration
    
    def clear_records(self):
        self.start_time_list.clear()
        self.end_time_list.clear()
