import sqlite3

def init_db():
    conn = sqlite3.connect('hospital_mock.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS ehr_data (
            patient_id TEXT PRIMARY KEY,
            blood_pressure_sys INTEGER,
            blood_pressure_dia INTEGER,
            heart_rate INTEGER
        )
    ''')
    
    # Insert mock data if empty
    c.execute("SELECT COUNT(*) FROM ehr_data")
    if c.fetchone()[0] == 0:
        c.executemany("INSERT INTO ehr_data VALUES (?, ?, ?, ?)", [
            ('p1', 120, 80, 72),
            ('p2', 130, 85, 75),
            ('p3', 115, 75, 68),
            ('p4', 140, 90, 80),
            ('p5', 125, 82, 70),
        ])
    conn.commit()
    conn.close()

def get_training_data():
    conn = sqlite3.connect('hospital_mock.db')
    c = conn.cursor()
    c.execute("SELECT blood_pressure_sys, blood_pressure_dia, heart_rate FROM ehr_data")
    data = c.fetchall()
    conn.close()
    return data
    
if __name__ == '__main__':
    init_db()
