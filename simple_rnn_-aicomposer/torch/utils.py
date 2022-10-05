import numpy as np
from pretty_midi import Instrument, Note, instrument_name_to_program


# note: pitch step duration

def GetNoteSequence(instrument: Instrument) -> np.ndarray:
    sorted_notes = sorted(instrument.notes, key=lambda x: x.start)
    assert len(sorted_notes) > 0
    notes = []
    prev_start = sorted_notes[0].start
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
