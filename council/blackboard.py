"""shared scratchpad every agent in a council can read and write."""

import threading


class Blackboard:
    def __init__(self):
        self._store = {}
        self._notes = []
        self._lock = threading.Lock()

    def write(self, key, value, author=""):
        with self._lock:
            self._store[key] = value
            self._notes.append((author, "write", key))

    def append(self, key, value, author=""):
        """append to a list stored under key, creating it if needed."""
        with self._lock:
            cur = self._store.get(key, [])
            if not isinstance(cur, list):
                cur = [cur]
            cur.append(value)
            self._store[key] = cur
            self._notes.append((author, "append", key))

    def read(self, key, default=None):
        with self._lock:
            return self._store.get(key, default)

    def keys(self):
        with self._lock:
            return sorted(self._store.keys())

    def note(self, text, author=""):
        with self._lock:
            self._notes.append((author, "note", text))

    def history(self):
        with self._lock:
            return list(self._notes)

    def snapshot(self):
        with self._lock:
            return dict(self._store)

    def dump_markdown(self):
        lines = ["# blackboard"]
        for key in self.keys():
            lines.append("## %s" % key)
            val = self.read(key)
            if isinstance(val, list):
                for item in val:
                    lines.append("- %s" % item)
            else:
                lines.append(str(val))
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"
