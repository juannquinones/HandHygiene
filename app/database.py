import sqlite3
from datetime import datetime

# Step 1: Connect to the SQLite database (or create it if it doesn't exist)


# Step 2: Create the table with 8 columns if it doesn't exist

#cursor.execute('''
#    CREATE TABLE IF NOT EXISTS my_table (
#        id INTEGER PRIMARY KEY AUTOINCREMENT,
#        date_time TEXT,
#        Step_1 REAL,
#        Step_2 REAL,
#        Step_3 REAL,
#        Step_4 REAL,
#        Step_5 REAL,
#        Step_6 REAL, 
#        Step_7 REAL
#    )
#''')
#conn.commit()

# Step 3: Define the function to add a record
def get_id(values):
    conn = sqlite3.connect('HandHygiene_database.db')
    cursor = conn.cursor()
    # Check if the input vector has exactly 7 elements
    if len(values) != 7:
        raise ValueError("The input vector must contain exactly 7 float numbers.")
    
    # Get the current date and time
    date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Insert the new record into the table
    cursor.execute('''
        INSERT INTO my_table (date_time, Step_1, Step_2, Step_3, Step_4, Step_5, Step_6, Step_7)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (date_time, *values))
    
    # Commit the transaction
    conn.commit()
    last_id = cursor.lastrowid
    conn.close()
    
    # Return the ID of the last inserted row
    return  last_id
