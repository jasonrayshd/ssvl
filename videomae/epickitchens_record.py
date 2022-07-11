from datetime import timedelta
import time


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def segment_name(self):
        return NotImplementedError()

    @property
    def participant(self):
        return NotImplementedError()

    @property
    def untrimmed_video_name(self):
        return NotImplementedError()

    @property
    def start_frame(self):
        return NotImplementedError()

    @property
    def end_frame(self):
        return NotImplementedError()

    @property
    def num_frames(self):
        return NotImplementedError()

    @property
    def label(self):
        return NotImplementedError()


def timestamp_to_sec(timestamp):
    x = time.strptime(timestamp, '%H:%M:%S.%f')
    sec = float(timedelta(hours=x.tm_hour,
                          minutes=x.tm_min,
                          seconds=x.tm_sec).total_seconds()) + float(
        timestamp.split('.')[-1]) / 100
    return sec


class EpicKitchensVideoRecord(VideoRecord):
    def __init__(self, tup):
        self._index = str(tup[0])
        self._series = tup[1]

    @property
    def participant(self):
        return self._series['participant_id']

    @property
    def untrimmed_video_name(self):
        return self._series['video_id']

    # has problem when reading P01_102_47 from csv annotation file
    @property
    def start_frame(self):
        return int(round(timestamp_to_sec(self._series['start_timestamp']) * self.fps))
    # has problem when reading P01_102_47
    @property
    def end_frame(self):
        return int(round(timestamp_to_sec(self._series['stop_timestamp']) * self.fps))

    @property
    def fps(self):
        is_100 = len(self.untrimmed_video_name.split('_')[1]) == 3
        return 50 if is_100 else 60

    @property
    def num_frames(self):
        return self.end_frame - self.start_frame

    @property
    def label(self):
        return {'verb': self._series['verb_class'] if 'verb_class' in self._series else -1,
                'noun': self._series['noun_class'] if 'noun_class' in self._series else -1}

    @property
    def metadata(self):
        return {'narration_id': self._index}