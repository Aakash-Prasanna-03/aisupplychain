import sqlite3, json
DB="research.db"
def init_db():
    with sqlite3.connect(DB) as c: c.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, payload TEXT)")
def save_run(id,payload):
    with sqlite3.connect(DB) as c: c.execute("INSERT OR REPLACE INTO runs VALUES (?,?)",(id,json.dumps(payload)))
