import numpy as np
from pretty_midi import Instrument, Note, instrument_name_to_program


# note: pitch step duration

# instrument.notes  该乐器的音符列表
# note.start 		开始时间
# note.end 		    结束时间
# note.pitch		音高
# note.velocity 	音符力度

def GetNoteSequence(instrument: Instrument) -> np.ndarray:
    sorted_notes = sorted(instrument.notes, key=lambda x: x.start)  # 将音符按照开始的时间进行排序
    assert len(sorted_notes) > 0
    notes = []
    prev_start = sorted_notes[0].start
    # 每个音符的音高, 距离上一个音符开始的时间, 持续时间
    for note in sorted_notes:
        notes.append([note.pitch, note.start -
                     prev_start, note.end-note.start])
        prev_start = note.start
    return np.array(notes)


def CreateMIDIInstrumennt(notes: np.ndarray, instrument_name: str) -> Instrument:
    instrument = Instrument(instrument_name_to_program(instrument_name))
    prev_start = 0
    for note in notes:
        prev_start += note[1]
        note = Note(start=prev_start, end=prev_start +
                    note[2], pitch=note[0], velocity=100)
        instrument.notes.append(note)
    return instrument
