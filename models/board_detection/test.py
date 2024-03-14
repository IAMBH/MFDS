import pymssql
from decouple import config


SERVER = config('SERVER')
USERNAME = config('USERNAME')
PASSWORD = config('PASSWORD')
DATABASE = config('DATABASE')

print(SERVER,USERNAME,PASSWORD, DATABASE)
print(type(SERVER))

conn = pymssql.connect(SERVER, USERNAME, PASSWORD, DATABASE)