
from database import test_connection

try:
    test_connection()
    print("MySQL connection successful!")
except Exception as error:
    print("MySQL connection failed:")
    print(type(error).__name__, error)
    print("MySQL connection failed:")
    print(type(error).__name__, error)