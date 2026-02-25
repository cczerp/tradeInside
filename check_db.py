import sqlite3
conn = sqlite3.connect('tradeinsider.db')
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM corporate_events")
print(f"Events in database: {cursor.fetchone()[0]}")
cursor.execute("SELECT * FROM corporate_events LIMIT 5")
for row in cursor.fetchall():
    print(row)
conn.close()