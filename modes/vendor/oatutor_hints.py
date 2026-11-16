"""Sequential hints using OATutor's dependency-unlock rule (MIT).

is_locked is adapted from CAHLR/OATutor SubHintSystem.isLocked, revision
939eb0e30d8ae2389c8b096c8cd522713a79090e. JS callbacks become a pure Python
function. reveal_next is project glue for a short linear hint sequence.
Copyright (c) 2023 Zachary A. Pardos (@zpardos) - CAHL research lab.
Full license: OATUTOR_LICENSE in this directory.
"""


def is_locked(hint_num, dependencies, hint_status):
    if hint_num == 0:
        return False
    return not all(hint_status[dependency] == 1 for dependency in dependencies)


def reveal_next(hints, status):
    status = list(status) if status else [0] * len(hints)
    for i, hint in enumerate(hints):
        if not status[i] and not is_locked(i, range(i), status):
            status[i] = 1
            return hint, status
    return "", status
