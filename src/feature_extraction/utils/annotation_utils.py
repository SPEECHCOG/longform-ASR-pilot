#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniil Kocharov (dan_ya)
"""
from copy import deepcopy
from bisect import bisect_right
from pathlib import Path
from typing import Union, Any

__all__ = ['Interval', 'Point', 'IntervalTier', 'PointTier', 'TextGrid']


def seg_name2id(name: str) -> int:
    result = 1
    levels = 'gbry'
    name = name.lower()
    idx_a = levels.find(name[0])
    if idx_a == -1:
        return result
    pos = int(name[1])
    result = 2 ** ((idx_a + 1) + (pos - 1) * len(levels) - 1)
    return result


class Interval:
    """
    Represents a labeled interval in time.
    # Interval
    >>> foo = Interval(3.0, 4.0, 'foo')
    >>> print(foo)
    Interval(3.0, 4.0, foo)
    >>> print(Interval())
    Interval(0.0, 0.0, None)

    # properties
    >>> foo.start_time
    3.0
    >>> foo.end_time
    4.0
    >>> foo.text
    'foo'
    >>> foo.start_time = 3.0
    >>> foo.end_time = 4.0
    >>> foo.text = 'foo'

    # Interval comparison with Interval, Point or digit (int/float)
    >>> Interval(3.5, 4.0) in foo
    True
    >>> Interval(3.5, 4.5) in foo
    False
    >>> Point(3.5) in foo
    True
    >>> 3.5 in foo
    True
    >>> Interval(3.0, 4.0) < Interval(4.0, 5.0)
    True
    >>> Interval(3.0, 4.0) < Point(4.0)
    False
    >>> Interval(3.0, 4.0) < Point(5.0)
    True
    >>> Interval(3.0, 4.0) < 4.0
    False
    >>> Interval(3.0, 4.0) < 5
    True
    >>> Interval(3.0, 4.0) <= Interval(3.5, 5.0)
    True
    >>> Interval(3.0, 4.0) <= 4.0
    True
    >>> Interval(3.0, 4.0, 'foo') == Interval(3.0, 4.0, 'bar')
    True
    # addition and substraction of Intervals and digits
    >>> Interval(3.0, 4.0) + Interval(5.0, 6.0)
    Interval(3.0, 6.0, None)
    >>> foo += Interval(3.0, 4.0)
    Interval(3.0, 4.0, foo)
    >>> foo += Interval(3.0, 6.0)
    Interval(3.0, 6.0, foo)
    >>> foo += Interval(1.0, 2.0)
    Interval(1.0, 6.0, foo)
    >>> Interval(3.0, 4.0) + 1
    Interval(4.0, 5.0, None)
    >>> 1 + Interval(3.0, 4.0)
    Interval(4.0, 5.0, None)
    >>> Interval(3.0, 5.0) - 1
    Interval(2.0, 4.0, None)
    >>> foo -= 1
    Interval(0.0, 5.0, None)

    # methods
    >>> foo.duration()
    5.0
    >>> foo.overlaps(Interval(4.5, 5.5))
    True
    >>> foo.overlaps(Interval(5.0, 5.5))
    False
    >>> foo.get_overlap(Interval(4.5, 5.5))
    0.5
    >>> foo.bounds()
    (0.0, 5.0)
    >>> foo.to_dict()
    {'start_time': 0.0, 'end_time': 5.0, 'text': 'foo'}
    """
    def __init__(self, start_time: float = 0.0, end_time: float = 0.0, text: str = ''):
        if start_time < 0:
            raise ValueError(f'The time position cannot be less than 0: {start_time}')
        if end_time < 0:
            raise ValueError(f'The time position cannot be less than 0: {end_time}')
        self._text = text
        self._start_time = start_time
        self._end_time = end_time
        if self._start_time > self._end_time:
            self._start_time, self._end_time = self._end_time, self._start_time

    @property
    def start_time(self) -> float:
        return self._start_time

    @start_time.setter
    def start_time(self, other: float):
        if other < 0:
            raise ValueError(f'The time position cannot be less than 0: {other}')
        if other > self._end_time:
            raise ValueError(f'Start time {other} cannot be larger than an end time {self._end_time}')
        self._start_time = other

    @property
    def end_time(self) -> float:
        return self._end_time

    @end_time.setter
    def end_time(self, other):
        if other < 0:
            raise ValueError(f'The time position cannot be less than 0: {other}')
        if other < self._start_time:
            raise ValueError(f'End time {other} cannot be smaller than an start time {self._start_time}')
        self._end_time = other

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, other: str):
        self._text = other

    def __repr__(self):
        text = self.text if self.text else None
        return f'Interval({self.start_time}, {self.end_time}, {text})'

    def __contains__(self, other: Union['Interval', 'Point', int, float]):
        if hasattr(other, 'start_time') and hasattr(other, 'end_time'):
            return self.start_time <= other.start_time and other.end_time <= self.end_time
        else:
            return self.start_time <= other <= self.end_time

    def __lt__(self, other: Union['Interval', 'Point', int, float]):
        if hasattr(other, 'start_time'):
            return self.end_time <= other.start_time  # (1,3) is less than (3,4)
        else:
            return self.end_time < other

    def __gt__(self, other: Union['Interval', 'Point', int, float]):
        if hasattr(other, 'end_time'):
            return other.end_time <= self.start_time
        else:
            return other < self.start_time

    def __le__(self, other: Union['Interval', 'Point', int, float]):
        if hasattr(other, 'end_time'):
            return self.end_time <= other.end_time
        else:
            return self.end_time <= other

    def __ge__(self, other: Union['Interval', 'Point', int, float]):
        if hasattr(other, 'start_time'):
            return other.start_time <= self.start_time
        else:
            return other <= self.start_time

    def __eq__(self, other: 'Interval'):
        if hasattr(other, 'start_time') and hasattr(other, 'end_time'):
            if self.start_time == other.start_time and self.end_time == other.end_time:
                return True
        return False

    def __add__(self, other: Union['Interval', int, float]):
        if hasattr(other, 'start_time') and hasattr(other, 'end_time') and hasattr(other, 'text'):
            return Interval(min(self.start_time, other.start_time), max(self.end_time, other.end_time), self.text + other.text)
        else:
            return Interval(self.start_time + other, self.end_time + other, self.text)

    def __iadd__(self, other: Union['Interval', int, float]):
        if hasattr(other, 'start_time') and hasattr(other, 'end_time') and hasattr(other, 'text'):
            self._start_time = min(self.start_time, other.start_time)
            self._end_time = max(self.end_time, other.end_time)
            self.text = self.text + other.text
        else:
            self._start_time += other
            self._end_time += other
        return self

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other: Union[int, float]):
        return Interval(self.start_time - other, self.end_time - other, self.text)

    def __isub__(self, other: Union[int, float]):
        self.start_time = self.start_time - other
        self.end_time = self.end_time - other
        return self

    def concatenate(self, other: Union['Interval', int, float], delimiter: str = ''):
        if hasattr(other, 'start_time') and hasattr(other, 'end_time') and hasattr(other, 'text'):
            self._start_time = min(self.start_time, other.start_time)
            self._end_time = max(self.end_time, other.end_time)
            self.text = self.text + delimiter + other.text

    def duration(self) -> float:
        return self.end_time - self.start_time

    def overlaps(self, other: 'Interval') -> bool:
        return self.start_time < other.end_time and self.end_time > other.start_time

    def get_overlap(self, other: 'Interval') -> Union['Interval', None]:
        if not self.overlaps(other):
            return None
        start_time = max(other.start_time, self.start_time)
        end_time = min(other.end_time, self.end_time)
        return Interval(start_time, end_time)

    def bounds(self) -> tuple[float, float]:
        return self.start_time, self.end_time

    def to_dict(self) -> dict[str, Union[str, float]]:
        return {'text': self.text, 'start_time': self.start_time, 'end_time': self.end_time}


class Point(Interval):
    """
    Represents a labeled point in time.
    # Point
    >>> foo = Point(3.0, 'foo')
    >>> print(foo)
    Point(3.0, foo)
    >>> print(Point())
    Point(0.0, None)
    # properties
    >>> foo.time
    3.0
    >>> foo.text
    'foo'
    >>> foo.time = 3.0
    >>> foo.text = 'foo'
    >>> Point(1.0, 'foo') == Point(1.0, 'bar')
    True

    # Point comparison with Interval, Point or digit (int/float) see in Interval class documentation. It works the same way.

    # addition and substraction of Points and digits
    >>> Point(3.0) + 1
    Point(4.0, None)
    >>> 1 + Point(4.0)
    Point(4.0, None)
    >>> Point(3.0) - 1
    Point(2.0, None)
    >>> foo -= 1
    Point(2.0, None)
    >>> foo += 1
    Point(3.0, None)
    """
    def __init__(self, time: float = 0.0, text: str = ''):
        super(Point, self).__init__(time, time, text)

    @property
    def time(self):
        return self.start_time

    @time.setter
    def time(self, other: Union[float, int]):
        self._start_time = other
        self._end_time = other

    def __repr__(self):
        text = self.text if self.text else None
        return f'Point({self.time}, {text})'

    def __lt__(self, other: Union['Interval', 'Point', int, float]):
        if hasattr(other, 'start_time'):
            return self.time < other.start_time
        else:
            return self.time < other

    def __gt__(self, other: Union['Interval', 'Point', int, float]):
        if hasattr(other, 'end_time'):
            return other.end_time < self.time
        else:
            return other < self.time


class _Tier:
    def __init__(self, name: str = '', start_time: float = 0.0, end_time: float = 0.0):
        if start_time < 0:
            raise ValueError(f'The time position cannot be less than 0: {start_time}')
        if end_time < 0:
            raise ValueError(f'The time position cannot be less than 0: {end_time}')
        self._name = name
        self._start_time = start_time
        self._end_time = end_time
        self._objects = []

    @property
    def start_time(self) -> float:
        return self._start_time

    @start_time.setter
    def start_time(self, other: float):
        if other < 0:
            raise ValueError(f'The time position cannot be less than 0: {other}')
        if other > self._end_time:
            raise ValueError(f'Start time {other} cannot be larger than an end time {self._end_time}')
        self._start_time = other

    @property
    def end_time(self) -> float:
        return self._end_time

    @end_time.setter
    def end_time(self, other):
        if other < 0:
            raise ValueError(f'The time position cannot be less than 0: {other}')
        if other < self._start_time:
            raise ValueError(f'End time {other} cannot be smaller than an start time {self._start_time}')
        self._end_time = other

    @property
    def objects(self):
        return deepcopy(self._objects)

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, other: str):
        self._name = other

    def __str__(self):
        name = self.name if self.name else None
        return f'<Tier {name}, {len(self)} points>'

    def __repr__(self):
        return str(self)

    def __iter__(self):
        return iter(self._objects)

    def __len__(self):
        return len(self._objects)

    def __getitem__(self, i):
        return self._objects[i]


class PointTier(_Tier):
    """
    Represents a labeled PointTier containing Points.
    # PointTier
    >>> foo = PointTier('foo')
    <PointTier foo, 0 points>
    # reading PointTier from Praat Tier file
    >>> bar = PointTier.from_file(file_path, 'bar')
    <PointTier bar, XXX points>
    # reading PointTier from WaveAssistant Seg file
    >>> bar = PointTier.from_seg_file(file_path, 'bar')
    <PointTier bar, XXX points>
    # creating PointTier from IntervalTier
    >>> bar = PointTier.from_interval_tier(IntervalTier(), 'bar')
    <PointTier bar, 0 points>

    # properties
    >>> foo.start_time = 0.0
    >>> foo.end_time = 4.0
    >>> foo.name = 'foo'
    >>> foo.start_time
    0.0
    >>> foo.end_time
    4.0
    >>> foo.name
    'foo'

    # adding to, removing from, iterating through Points in PointTier
    >>> foo.add(0, 'first')
    <PointTier foo, 1 points>
    >>> foo.add_point(Point(1, 'second'))
    <PointTier foo, 2 points>
    >>> foo.end_time
    4.0
    >>> foo.add_point(Point(5, 'third'))
    <PointTier foo, 3 points>
    >>> foo.end_time
    5.0
    >>> foo.add_point(Point(10, 'forth'))
    <PointTier foo, 4 points>
    >>> len(foo)
    4
    >>> foo.add(10, 'last')
    >>> foo[3]
    Point(10.0, 'forth')
    >>> foo.add(10, 'last', overwrite=True)
    >>> foo[3]
    Point(10.0, 'last')
    >>> foo.remove(0.0, 'first')
    >>> foo.remove_point(Point(10.0, 'last'))
    >>> for p in foo:
    >>>     print(p)
    Point(1.0, 'second')
    Point(5.0, 'third')
    >>> foo.bounds()
    (1.0, 5.0)

    # reading from Praat Tier file
    >>> foo.read(file_path)
    >>> foo.read(file_path, encoding='utf-8')
    # reading from WaveAssistant Seg file
    >>> foo.read_seg(file_path)
    >>> foo.read_seg(file_path, encoding='cp1251')
    # writing to Praat Tier file
    >>> foo.write(file_path)
    >>> foo.write(file_path, encoding='utf-8')
    >>> foo.write(file_path, object_class='PitchTier', encoding='utf-8')
    # writing to WaveAssistant Seg file
    >>> foo.write_seg(file_path)
    >>> foo.write_seg(file_path, encoding='cp1251')
    >>> foo.write_seg(file_path, samplerate=22050)
    >>> foo.write_seg(file_path, samplerate=22050, byterate=2)
    """
    def __init__(self, name: str = '', start_time: float = 0.0, end_time: float = 0.0):
        super(PointTier, self).__init__(name, start_time, end_time)

    def __str__(self):
        name = self.name if self.name else None
        return f'<PointTier {name}, {len(self)} points>'

    def __repr__(self):
        return str(self)

    def add(self, time: float, text: str = '', overwrite: bool = False):
        self.add_point(Point(time, text), overwrite=overwrite)

    def add_point(self, point: Point, overwrite: bool = False):
        self.start_time = min(self.start_time, point.time)
        self.end_time = max(self.end_time, point.time)
        i = bisect_right(self._objects, point)
        if i < len(self._objects) and self._objects[i].time == point.time:
            if overwrite:
                self[i].text = point.text
            else:
                return
        elif i > 0 and i == len(self._objects) and self._objects[-1].time == point.time:
            if overwrite:
                self[-1].text = point.text
            else:
                return
        else:
            self._objects.insert(i, point)

    def bounds(self) -> tuple[float, float]:
        return self.start_time, self.end_time

    def remove(self, time: float, text: str):
        self.remove_point(Point(time, text))

    def remove_point(self, point: Point):
        self._objects.remove(point)

    def read(self, file: Union[str, Path], encoding: str = 'utf-8'):
        """
        Read the Tier contained in the Praat PointTier
        """
        with open(file, 'r', encoding=encoding) as source:
            for i in range(3):
                source.readline()
            self.start_time = float(source.readline().split(' = ')[-1].strip())
            self.end_time = float(source.readline().split(' = ')[-1].strip())
            tier_size = int(source.readline().split(' = ')[-1].strip())
            for i in range(tier_size):
                source.readline()
                time = float(source.readline().split(' = ')[-1].strip())
                text = source.readline().split(' = ')[-1].strip().strip('"')
                self.add(time, text)

    def read_seg(self, file: Union[str, Path], encoding: str = 'cp1251'):
        """
        Read the Tier contained in the Seg-formatted file
        """
        with open(file, 'r', encoding=encoding) as source:
            source.readline()  # [PARAMETERS]
            samplerate = float(source.readline().split('=')[-1])  # SAMPLING_FREQ
            byterate = float(source.readline().split('=')[-1])  # BYTE_PER_SAMPLE
            source.readline()  # CODE
            source.readline()  # N_CHANNEL
            source.readline()  # N_LABEL
            source.readline()  # [LABELS]
            tier_name = ''
            for line in source:
                parts = line.strip().split(',')
                time = float(parts[0]) / byterate / samplerate
                tier_name = parts[1]
                text = ','.join(parts[2:])
                self.add(time, text)
            if self.name == '':
                self.name = tier_name

    def write(self, file: Union[str, Path], object_class: str = 'TextTier', encoding: str = 'utf-8'):
        with open(file, 'w', encoding=encoding) as fout:
            fout.write('File type = "ooTextFile"\n')
            fout.write(f'Object class = "{object_class}"\n\n')
            fout.write('Object class = "TextTier"\n\n')
            fout.write(f'xmin = {self.start_time}\n')
            fout.write(f'xmax = {self.end_time}\n')
            fout.write(f'points: size = {len(self)}\n')
            for (i, point) in enumerate(self._objects, 1):
                fout.write(f'points [{i}]:\n')
                fout.write(f'\ttime = {point.time}\n')
                fout.write(f'\ttext = {point.text}\n')

    def write_seg(self, file: Union[str, Path], tier_type: str = 'G1', samplerate: int = 22050, byterate: int = 2, encoding: str = 'cp1251'):
        with open(file, 'w', encoding=encoding) as fout:
            if tier_type is not None:
                tier_type = seg_name2id(tier_type)
            else:
                tier_type = 0
            fout.write('[PARAMETERS]\n')
            fout.write(f'SAMPLING_FREQ={samplerate}\n')
            fout.write(f'BYTE_PER_SAMPLE={byterate}\n')
            fout.write('CODE=0\n')
            fout.write('N_CHANNEL=1\n')
            fout.write(f'N_LABEL={len(self)}\n')
            fout.write('[LABELS]\n')
            for point in self._objects:
                value = round(point.time * samplerate * byterate)
                fout.write(f'{value},{tier_type},{point.text}\n')

    # alternative constructors
    @classmethod
    def from_file(cls, file: Union[str, Path], name: str = '') -> 'PointTier':
        pt = cls(name=name)
        pt.read(file)
        return pt

    @classmethod
    def from_seg_file(cls, file: Union[str, Path], name: str = '') -> 'PointTier':
        pt = cls(name=name)
        pt.read_seg(file)
        return pt

    @classmethod
    def from_interval_tier(cls, tier: 'IntervalTier', name: str = '') -> 'PointTier':
        pt = cls(name=name)
        if pt.name == '':
            pt.name = tier.name
        for interval in tier:
            pt.add(interval.start_time, interval.text, overwrite=True)
            pt.add(interval.end_time, '', overwrite=True)
        return pt


class IntervalTier(_Tier):
    """
    Represents a labeled IntervalTier containing Intervals.
    # IntervalTier
    >>> foo = IntervalTier('foo')
    <IntervalTier foo, 0 intervals>
    # reading IntervalTier from Praat Tier file
    >>> bar = IntervalTier.from_file(file_path, 'bar')
    <IntervalTier bar, XXX intervals>
    # reading IntervalTier from WaveAssistant Seg file
    >>> bar = IntervalTier.from_seg_file(file_path, 'bar')
    <IntervalTier bar, XXX intervals>
    # creating IntervalTier from PointTier
    >>> bar = IntervalTier.from_point_tier(PointTier(), 'bar')
    <IntervalTier bar, 0 intervals>
    # creating IntervalTier from list of Point
    >>> bar = IntervalTier.from_points([Point(1.0, 'foo'), Point(2.0, 'bar'), Point(3.0)], 'bar')
    <IntervalTier bar, 2 intervals>

    # properties
    >>> bar.name = 'foo'
    >>> bar.start_time
    1.0
    >>> bar.end_time
    3.0
    >>> bar.name
    'foo'

    # adding to, removing from, iterating through Intervals in IntervalTier
    >>> foo.add(0, 1, 'first')
    <IntervalTier foo, 1 intervals>
    >>> foo.add_interval(Interval(1, 2, 'second'))
    <IntervalTier foo, 2 intervals>
    >>> foo.end_time
    2.0
    >>> foo.add_point(Interval(4, 5, 'third'))
    <IntervalTier foo, 3 intervals>
    >>> foo.end_time
    5.0
    >>> foo.add_point(Interval(9, 10, 'forth'))
    <IntervalTier foo, 4 intervals>
    >>> len(foo)
    4
    >>> foo[2]
    Interval(1.0, 2.0, 'second')
    >>> foo.add(9, 10, 'last')
    >>> foo[-1]
    Interval(9.0, 10.0, 'forth')
    >>> foo.add(9, 10, 'last', overwrite=True)
    >>> foo[-1]
    Interval(9.0, 10.0, 'last')
    >>> foo.remove(0, 1, 'first')
    >>> foo.remove_interval(Interval(9, 10, 'last'))
    >>> for i in foo:
    >>>     print(i)
    Interval(1.0, 2.0, 'second')
    Interval(4.0, 5.0, 'third')
    >>> foo.to_dict()
    [{'start_time': 1.0, 'end_time': 2.0, 'text': 'second'}, {'start_time': 4.0, 'end_time': 5.0, 'text': 'third'}]
    >>> foo.bounds()
    (1.0, 5.0)

    # reading from Praat Tier file
    >>> foo.read(file_path)
    >>> foo.read(file_path, encoding='utf-8')
    # writing to Praat Tier file
    >>> foo.write(file_path)
    >>> foo.write(file_path, encoding='utf-8')
    # writing to WaveAssistant Seg file
    >>> foo.write_seg(file_path)
    >>> foo.write_seg(file_path, encoding='cp1251')
    >>> foo.write_seg(file_path, samplerate=22050)
    >>> foo.write_seg(file_path, samplerate=22050, byterate=2)
    """
    def __init__(self, name: str = '', start_time: float = 0.0, end_time: float = 0.0):
        super(IntervalTier, self).__init__(name, start_time, end_time)

    def __str__(self):
        name = self.name if self.name else None
        return f'<IntervalTier {name}, {len(self)} intervals>'

    def __repr__(self):
        return str(self)

    def add(self, start_time: float, end_time: float, text: str = '', overwrite: bool = False):
        self.add_interval(Interval(start_time, end_time, text), overwrite=overwrite)

    def add_interval(self, interval: Interval, overwrite: bool = False):
        self.start_time = min(self.start_time, interval.start_time)
        self.end_time = max(self.end_time, interval.end_time)
        i = bisect_right(self._objects, interval)
        if i < len(self._objects) and self._objects[i] == interval:
            if overwrite:
                self[i].text = interval.text
            else:
                return
        elif i > 0 and i == len(self._objects) and self._objects[-1] == interval:
            if overwrite:
                self[-1].text = interval.text
            else:
                return
        else:
            self._objects.insert(i, interval)

    def bounds(self) -> tuple[float, float]:
        return self.start_time, self.end_time

    def remove(self, start_time: float, end_time: float, text: str):
        self.remove_interval(Interval(start_time, end_time, text))

    def remove_interval(self, interval: Interval):
        self._objects.remove(interval)

    def to_dict(self) -> list[dict[str, Union[str, float]]]:
        return [u.to_dict() for u in self]

    def read(self, file: Union[str, Path], encoding: str = 'utf-8'):
        """
        Read the Tier contained in the Praat IntervalTier
        """
        with open(file, 'r', encoding=encoding) as source:
            for i in range(3):
                source.readline()
            self.start_time = float(source.readline().split(' = ')[-1].strip())
            self.end_time = float(source.readline().split(' = ')[-1].strip())
            tier_size = int(source.readline().split(' = ')[-1].strip())
            for i in range(tier_size):
                source.readline()
                start_time = float(source.readline().split(' = ')[-1].strip())
                end_time = float(source.readline().split(' = ')[-1].strip())
                text = source.readline().split(' = ')[-1].strip().strip('"')
                self.add(start_time, end_time, text)

    def write(self, file: Union[str, Path], encoding: str = 'utf-8'):
        with open(file, 'w', encoding=encoding) as fout:
            fout.write('File type = "ooTextFile"\n')
            fout.write('Object class = "IntervalTier"\n\n')
            fout.write(f'xmin = {self.start_time}\n')
            fout.write(f'xmax = {self.end_time}\n')
            output = self._fill_in_the_gaps()
            fout.write(f'intervals: size = {len(output)}\n')
            for (i, interval) in enumerate(output, 1):
                fout.write(f'intervals [{i}]:\n')
                fout.write(f'\txmin = {interval.start_time}\n')
                fout.write(f'\txmax = {interval.end_time}\n')
                fout.write(f'\ttext = {interval.text}\n')

    def write_seg(self, file: Union[str, Path], tier_type: [str, None] = None, samplerate: int = 22050, byterate: int = 2, encoding: str = 'cp1251'):
        pt = PointTier.from_interval_tier(self)
        pt.write_seg(file, tier_type=tier_type, samplerate=samplerate, byterate=byterate, encoding=encoding)

    def _fill_in_the_gaps(self) -> list[Interval]:
        prev_t = self.start_time
        output = []
        for interval in self:
            if prev_t < interval.start_time:
                output.append(Interval(prev_t, interval.start_time))
            output.append(interval)
            prev_t = interval.end_time
        if len(output) != 0 and output[-1].end_time < self.end_time:
            output.append(Interval(output[-1].end_time, self.end_time))
        return output

    # alternative constructors
    @classmethod
    def from_file(cls, file: Union[str, Path], name: str = '') -> 'IntervalTier':
        it = cls(name=name)
        it.read(file)
        return it

    @classmethod
    def from_seg_file(cls, file: Union[str, Path], name: str = '') -> 'IntervalTier':
        pt = PointTier.from_seg_file(file, name)
        it = IntervalTier.from_point_tier(pt)
        return it

    @classmethod
    def from_points(cls, points: Union[list[Point], PointTier], name: str = '') -> 'IntervalTier':
        it = cls(name=name)
        for i in range(1, len(points)):
            it.add(points[i - 1].time, points[i].time, points[i - 1].text)
        return it

    @classmethod
    def from_point_tier(cls, tier: PointTier, name: str = '') -> 'IntervalTier':
        it = cls(name=name)
        if it.name == '':
            it.name = tier.name
        for i in range(1, len(tier)):
            it.add(tier[i - 1].time, tier[i].time, tier[i - 1].text)
        return it


class TextGrid(_Tier):
    """
    Represents a TextGrid.
    # TextGrid
    >>> foo = TextGrid('foo')
    <TextGrid foo, 0 tiers>
    # reading TextGrid from Praat TextGrid file
    >>> bar = TextGrid.from_file(file_path, 'bar')
    <TextGrid bar, 0 tiers>
    >>> bar = bar.append(IntervalTier('it_foo'))
    <TextGrid bar, 1 tiers>
    >>> bar = bar.extend([PointTier('pt_foo'), IntervalTier('it_bar'), PointTier('pt_bar')])
    <TextGrid bar, 4 tiers>
    >>> bar.get_tier_names()
    ['it_foo', 'pt_foo', 'it_bar', 'pt_bar']
    >>> 'it' in bar
    False
    >>> 'it_foo' in bar
    True
    >>> del bar['it_foo']  # delete all Tiers with name 'it_foo'
    ['pt_foo', 'it_bar', 'pt_bar']
    >>> bar.insert(0, IntervalTier('it_foo'))
    ['it_foo', 'pt_foo', 'it_bar', 'pt_bar']

    >>> bar.index('pt_foo')
    1
    
    >>> bar[0]
    <IntervalTier it_foo, 0 intervals>
    >>> bar['pt_foo']
    <PointTier pt_foo, 0 points>

    # properties
    >>> bar.name = 'foo'
    >>> bar.start_time
    0.0
    >>> bar.end_time
    0.0
    >>> bar.name
    'foo'

    # reading and writing files
    # reading from Praat TextGrid file
    >>> foo.read(file_path)
    >>> foo.read(file_path, encoding='utf-8')
    # writing to Praat TextGrid file
    >>> foo.write(file_path)
    >>> foo.write(file_path, encoding='utf-8')
    """
    def __init__(self, name: str = '', start_time: float = 0.0, end_time: float = 0.0):
        super(TextGrid, self).__init__(name, start_time, end_time)

    def __str__(self):
        return f'<TextGrid {self.name}, {len(self)} Tiers>'

    def __repr__(self):
        return str(self)

    def __contains__(self, value: str):
        for t in self:
            if t.name == value:
                return True
        return False

    def __delitem__(self, i: [str, int]):
        if type(i) is str:
            to_remove = [t for t in range(len(self._objects) - 1, -1, -1) if self._objects[t].name == i]
            for t in to_remove:
                del self._objects[t]
        elif type(i) is int:
            del self._objects[i]
        else:
            raise TypeError(f'The index type should be either str or int.')
        return None

    def __getitem__(self, i: [str, int]):
        if type(i) is str:
            result = None
            for t in self._objects:
                if t.name == i:
                    return t
        elif type(i) is int:
            result = self._objects[i]
        else:
            raise TypeError(f'The index type should be either str or int.')
        return result

    def get_tier_names(self) -> list[str]:
        return [tier.name for tier in self]

    def append(self, tier: Union[PointTier, IntervalTier], name: Union[str, None] = None):
        self.start_time = min(self.start_time, tier.start_time)
        self.end_time = max(self.end_time, tier.end_time)
        if name is not None:
            tier = deepcopy(tier)
            tier.name = name
        self._objects.append(tier)

    def extend(self, tiers: list[Union[PointTier, IntervalTier]]):
        for tier in tiers:
            self.append(tier)

    def index(self, name: str) -> Union[int, None]:
        for i, tier in enumerate(self._objects):
            if getattr(tier, 'name', None) == name:
                return i
        return None

    def insert(self, index: int, tier: Union['IntervalTier', 'PointTier']):
        if not is_tier(tier):
            raise TypeError("Only IntervalTier or PointTier instances can be inserted.")
        if not isinstance(index, int):
            raise TypeError("Index must be an integer.")
        if index < 0 or index > len(self._objects):
            raise IndexError("Index out of bounds.")
        self._objects.insert(index, tier)

    def read(self, file: Union[str, Path], encoding: str = 'utf-8'):
        if self.name == '':
            self.name = Path(file).stem
        with open(file, 'r', encoding=encoding) as source:
            for i in range(3):
                source.readline()
            self.start_time = float(source.readline().split(' = ')[-1].strip())
            self.end_time = float(source.readline().split(' = ')[-1].strip())
            source.readline()
            n_tiers = int(source.readline().strip().split()[-1])
            source.readline()
            for i_tier in range(n_tiers):
                source.readline()
                if source.readline().split(' = ')[-1].strip().strip('"') == 'IntervalTier':
                    tier_name = source.readline().split(' = ')[-1].strip().strip('"')
                    start_time = float(source.readline().split(' = ')[-1].strip())
                    end_time = float(source.readline().split(' = ')[-1].strip())
                    tier = IntervalTier(tier_name, start_time, end_time)
                    tier_size = int(source.readline().split(' = ')[-1].strip())
                    for j in range(tier_size):
                        source.readline()
                        start_time = float(source.readline().split(' = ')[-1].strip())
                        end_time = float(source.readline().split(' = ')[-1].strip())
                        text = source.readline().split(' = ')[-1].strip().strip('"')
                        tier.add(start_time, end_time, text)
                    self.append(tier)
                else:
                    tier_name = source.readline().split(' = ')[-1].strip().strip('"')
                    start_time = float(source.readline().split(' = ')[-1].strip())
                    end_time = float(source.readline().split(' = ')[-1].strip())
                    tier = PointTier(tier_name, start_time, end_time)
                    tier_size = int(source.readline().split(' = ')[-1].strip())
                    for j in range(tier_size):
                        source.readline()
                        time = float(source.readline().split(' = ')[-1].strip())
                        text = source.readline().split(' = ')[-1].strip().strip('"')
                        tier.add(time, text)
                    self.append(tier)
        return self

    def write(self, file: Union[str, Path], encoding: str = 'utf-8'):
        with open(file, 'w', encoding=encoding) as fout:
            fout.write('File type = "ooTextFile"\n')
            fout.write('Object class = "TextGrid"\n\n')
            fout.write(f'xmin = {self.start_time}\n')
            fout.write(f'xmax = {self.end_time}\n')
            fout.write('tiers? <exists>\n')
            fout.write(f'size = {len(self)}\n')
            fout.write('item []:\n')
            for (i, tier) in enumerate(self, 1):
                fout.write(f'\titem [{i}]:\n')
                if type(tier) == IntervalTier:
                    fout.write('\t\tclass = "IntervalTier"\n')
                    fout.write(f'\t\tname = "{tier.name}"\n')
                    fout.write(f'\t\txmin = {tier.start_time}\n')
                    fout.write(f'\t\txmax = {tier.end_time}\n')
                    output = tier._fill_in_the_gaps()
                    fout.write(f'\t\tintervals: size = {len(output)}\n')
                    for (j, interval) in enumerate(output, 1):
                        fout.write(f'\t\t\tintervals [{j}]:\n')
                        fout.write(f'\t\t\t\txmin = {interval.start_time}\n')
                        fout.write(f'\t\t\t\txmax = {interval.end_time}\n')
                        fout.write(f'\t\t\t\ttext = "{interval.text}"\n')
                elif type(tier) == PointTier:
                    fout.write('\t\tclass = "TextTier"\n')
                    fout.write(f'\t\tname = "{tier.name}"\n')
                    fout.write(f'\t\txmin = {tier.start_time}\n')
                    fout.write(f'\t\txmax = {tier.end_time}\n')
                    fout.write(f'\t\tpoints: size = {len(tier)}\n')
                    for (j, point) in enumerate(tier, 1):
                        fout.write(f'\t\t\tpoints [{j}]:\n')
                        fout.write(f'\t\t\t\ttime = {point.time}\n')
                        fout.write(f'\t\t\t\ttext = "{point.text}"\n')

    @classmethod
    def from_file(cls, file: Union[str, Path], name: str = ''):
        tg = cls(name=name)
        tg.read(file)
        return tg

    @classmethod
    def extract_selected(cls, file: Union[str, Path], start_time: float, end_time: float, name: str = ''):
        tg = cls.from_file(file)
        target_interval = Interval(start_time, end_time)
        extracted_tg = cls(name=name, start_time=start_time, end_time=end_time - start_time)
        for tier in tg:
            items_to_add = [i for i in tier if i in target_interval]
            if type(tier) is PointTier:
                extracted_tier = PointTier(name=tier.name)
                for item in items_to_add:
                    extracted_tier.add(item.time - start_time, item.text)
            else:
                extracted_tier = IntervalTier(name=tier.name)
                for item in items_to_add:
                    extracted_tier.add(item.start_time - start_time, item.end_time - start_time, item.text)
            extracted_tg.append(extracted_tier)
        return extracted_tg

def is_tier(obj: Any) -> bool:
    return isinstance(obj, (IntervalTier, PointTier))
